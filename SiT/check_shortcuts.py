"""Synthetic SiT shortcut checks; no image data or pretrained model required."""
import io
import subprocess
import types

import torch

from models import SiT, local_average


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    positions = torch.arange(16).reshape(4, 4)
    x = torch.randn(1, 16, 3, dtype=torch.float64, requires_grad=True)
    expected = []
    for row in range(4):
        for col in range(4):
            indices = positions[max(0, row-2):row+3, max(0, col-2):col+3].flatten()
            expected.append(x[:, indices].mean(1))
    torch.testing.assert_close(local_average(x), torch.stack(expected, dim=1))
    torch.testing.assert_close(local_average(torch.ones_like(x)), torch.ones_like(x))
    assert torch.autograd.gradcheck(local_average, (x,))
    source = subprocess.check_output(['git', 'show', '1954452:SiT/models.py'], text=True)
    source = source.replace('from flash_attn import flash_attn_qkvpacked_func, flash_attn_kvpacked_func', '')
    original = types.ModuleType('original_sit')
    exec(compile(source, 'original_sit', 'exec'), original.__dict__)
    def attention(qkv, dropout, scale):
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout, scale=scale).transpose(1, 2)
    original.flash_attn_qkvpacked_func = attention
    config = dict(input_size=8, hidden_size=32, depth=4, num_heads=4, num_classes=100, min_sigma=.8)
    old = original.SiT(**config).eval()
    # Nonzero learned branches make this more than an initialization check.
    with torch.no_grad():
        for p in old.parameters():
            if p.requires_grad:
                p.normal_(std=.03)
    new = SiT(**config, beta=0).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    inputs = torch.randn(2, 4, 8, 8)
    times, labels = torch.rand(2), torch.tensor([0, 99])
    torch.testing.assert_close(old(inputs, times, labels), new(inputs, times, labels), rtol=0, atol=0)
    count = sum(p.numel() for p in new.parameters() if p.requires_grad)
    new.set_method_settings(min_sigma=1, beta=0)
    control = new(inputs, times, labels)
    new.set_method_settings(min_sigma=1, beta=.5)
    torch.testing.assert_close(control, new(inputs, times, labels), rtol=0, atol=0)
    new.set_method_settings(min_sigma=.8, beta=.5)
    assert sum(p.numel() for p in new.parameters() if p.requires_grad) == count
    buffer = io.BytesIO()
    torch.save(dict(model=new.state_dict(), settings=new.method_settings()), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = SiT(**config)
    restored.set_method_settings(**saved['settings'])
    restored.load_state_dict(saved['model'])
    restored.eval()
    torch.testing.assert_close(restored(inputs, times, labels), new(inputs, times, labels), rtol=0, atol=0)
    loss = new(inputs, times, labels).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)
    print('PASS: grid positions, normalized borders, constants and average gradients')
    print('PASS: beta=0 original recovery, unit multipliers, unchanged parameters')
    print('PASS: checkpoint settings/outputs and finite full forward/backward')
    from feature_eval import FeatureAdapter
    class FakeVAE(torch.nn.Module):
        def encode(self, images):
            return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: images))
    class RecordingModel(torch.nn.Module):
        y_embedder = types.SimpleNamespace(num_classes=100)
        def forward_tokens(self, latents, times, labels):
            assert torch.equal(times, torch.ones_like(times))
            assert torch.equal(labels, torch.full_like(labels, 100))
            return latents.flatten(2).transpose(1, 2), None
    adapter = FeatureAdapter(RecordingModel(), FakeVAE()).eval()
    output = adapter.forward_feat(inputs)[0]
    torch.testing.assert_close(output, (inputs * .18215).flatten(2).transpose(1, 2))
    assert not adapter.feature_has_cls
    print('PASS: feature extraction uses clean t=1, null labels and all patch tokens')
    from feature_eval import extract_layers
    from torch.utils.data import TensorDataset
    class LayerModel(RecordingModel):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.enc_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(2)])
            self.dec_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(2)])
        def forward_tokens(self, values, times, labels):
            assert torch.equal(labels, torch.full_like(labels, 100))
            tokens = values.flatten(2).transpose(1, 2)
            for block in list(self.enc_blocks)+list(self.dec_blocks):
                tokens = block(tokens)
            return tokens, None
    moments = torch.randn(3, 8, 2, 2)
    dataset = TensorDataset(moments, torch.tensor([0, 99, 42]))
    model = LayerModel().eval()
    rng = torch.get_rng_state().clone()
    first = extract_layers(model, dataset, [0, 2], [1, 4], .5, 17)
    second = extract_layers(model, dataset, [0, 2], [1, 4], .5, 17)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    clean = extract_layers(model, dataset, [0, 2], [4], 1., 17)
    expected = (moments[[0, 2], :4]*.18215).flatten(2).mean(2)
    torch.testing.assert_close(clean[:, 0], expected)
    assert not model.enc_blocks[0]._forward_hooks
    print('PASS: layer collection, fixed noise, clean endpoint, null labels, RNG preservation and hook removal')


if __name__ == '__main__':
    main()
