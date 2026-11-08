"""Small synthetic checks: run with python MAE/check_shortcuts.py."""
import io
import subprocess
import types

import torch

from models_mae import MaskedAutoencoderViT, visible_local_average


def small_model(beta=0, min_decay=.7):
    return MaskedAutoencoderViT(
        img_size=32, patch_size=4, embed_dim=32, depth=4, num_heads=4,
        decoder_embed_dim=24, decoder_depth=2, decoder_num_heads=4,
        beta=beta, min_decay=min_decay)


def check_average(local_kernel='uniform5'):
    positions = torch.tensor([[24, 0, 12, 4, 20]])
    x = torch.randn(1, 5, 3, dtype=torch.float64, requires_grad=True)
    expected = []
    for position in positions[0]:
        row, col = divmod(position.item(), 5)
        neighbours = [(abs(p.item() // 5 - row) <= 2 and
                       abs(p.item() % 5 - col) <= 2) for p in positions[0]]
        weights = x.new_ones(5)
        if local_kernel == 'gaussian5':
            vector = x.new_tensor([1, 4, 6, 4, 1])
            weights = torch.stack([vector[p.item()//5-row+2]*vector[p.item()%5-col+2]
                                   if nearby else x.new_tensor(0.) for p, nearby in zip(positions[0], neighbours)])
        weights = weights[neighbours]
        expected.append((x[0, neighbours]*weights[:, None]).sum(0)/weights.sum())
    torch.testing.assert_close(visible_local_average(x, positions, (5, 5), local_kernel),
                               torch.stack(expected).unsqueeze(0))
    constant = torch.full_like(x, 3.5)
    torch.testing.assert_close(visible_local_average(constant, positions, (5, 5), local_kernel), constant)
    permutation = torch.tensor([2, 4, 0, 3, 1])
    torch.testing.assert_close(
        visible_local_average(x[:, permutation], positions[:, permutation], (5, 5), local_kernel),
        visible_local_average(x, positions, (5, 5), local_kernel)[:, permutation])
    assert torch.autograd.gradcheck(lambda value: visible_local_average(
        value, positions, (5, 5), local_kernel), (x,))
    print('PASS:', local_kernel, 'spatial neighbours, shuffled order, constants, average gradients')


def check_original():
    # Compare against the pre-change implementation using the same attention backend.
    source = subprocess.check_output(
        ['git', 'show', '344c2e8:MAE/models_mae.py'], text=True)
    original = types.ModuleType('original_mae')
    exec(compile(source, 'original_mae', 'exec'), original.__dict__)
    old = original.MaskedAutoencoderViT(
        img_size=32, patch_size=4, embed_dim=32, depth=4, num_heads=4,
        decoder_embed_dim=24, decoder_depth=2, decoder_num_heads=4, min_decay=.7)
    new = small_model()
    new.load_state_dict(old.state_dict(), strict=True)
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    images = torch.randn(2, 3, 32, 32)
    torch.manual_seed(9)
    before = old.forward_encoder(images, .75)
    torch.manual_seed(9)
    after = new.forward_encoder(images, .75)
    for a, b in zip(before[:4], after[:4]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.manual_seed(9)
    loss_old = old(images)[0]
    torch.manual_seed(9)
    torch.testing.assert_close(loss_old, new(images)[0], rtol=0, atol=0)
    torch.testing.assert_close(old.forward_feat(images)[0], new.forward_feat(images)[0], rtol=0, atol=0)
    new.set_method_settings(min_decay=.7, beta=0, local_kernel='gaussian5')
    torch.testing.assert_close(old.forward_feat(images)[0], new.forward_feat(images)[0], rtol=0, atol=0)
    active = small_model(beta=.5)
    active.load_state_dict(new.state_dict())
    assert sum(p.numel() for p in active.parameters() if p.requires_grad) == sum(
        p.numel() for p in new.parameters() if p.requires_grad)
    # Unit multipliers recover the same architecture with standard shortcuts.
    new.set_method_settings(min_decay=1, beta=0)
    active.set_method_settings(min_decay=1, beta=1, local_kernel='gaussian5')
    torch.testing.assert_close(new.forward_feat(images)[0], active.forward_feat(images)[0], rtol=0, atol=0)
    print('PASS: exact beta=0 recovery, unit-multiplier control, unchanged parameters')


def check_mask_and_checkpoint(local_kernel='uniform5'):
    model = small_model(beta=.5)
    model.set_method_settings(min_decay=.7, beta=.5, local_kernel=local_kernel)
    images = torch.randn(2, 3, 32, 32)
    torch.manual_seed(31)
    encoded = model.forward_encoder(images, .75)
    patches = model.patchify(images)
    patches[encoded[1].bool()] = torch.randn_like(patches[encoded[1].bool()]) * 100
    torch.manual_seed(31)
    changed = model.forward_encoder(model.unpatchify(patches), .75)
    torch.testing.assert_close(encoded[0], changed[0], rtol=0, atol=0)
    block = model.blocks[-1]
    x = torch.randn(2, 5, 32)
    positions = torch.tensor([[0, 2, 12, 63], [63, 12, 2, 0]])
    first = block.shortcut(x, block.decay[0], positions, (8, 8))
    torch.testing.assert_close(first[:, :1], x[:, :1] * block.decay[0])
    first = first + block.attn(block.norm1(x))
    expected = block.shortcut(first, block.decay[1], positions, (8, 8)) + block.mlp(block.norm2(first))
    torch.testing.assert_close(block(x, positions, (8, 8)), expected)
    model.eval()
    output = model.forward_feat(images)[0]
    buffer = io.BytesIO()
    torch.save(dict(model=model.state_dict(), method_settings=model.method_settings()), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = small_model()
    restored.set_method_settings(**saved['method_settings'])
    assert restored.method_settings() == model.method_settings()
    restored.load_state_dict(saved['model'])
    restored.eval()
    torch.testing.assert_close(output, restored.forward_feat(images)[0], rtol=0, atol=0)
    loss = model(images)[0]
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all()
                                      for p in model.parameters() if p.grad is not None)
    print('PASS:', local_kernel, 'hidden patches isolated, CLS, second average, checkpoint round trip, finite backward')


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.manual_seed(0)
    check_average()
    check_average('gaussian5')
    check_original()
    check_mask_and_checkpoint()
    check_mask_and_checkpoint('gaussian5')
