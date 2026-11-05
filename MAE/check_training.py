"""Synthetic checks for update counts, validation isolation and checkpoint resume."""
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from check_shortcuts import small_model
from engine_pretrain import train_one_epoch, evaluate
from main_pretrain import get_args_parser
from util import misc
from util.run_log import RunLog


class ToyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))

    def forward(self, samples, mask_ratio):
        return (samples * self.weight - 1).square().mean(), [], []


def main():
    torch.set_num_threads(2)
    args = get_args_parser().parse_args([
        '--device', 'cpu', '--temp5', '0', '--epochs', '1', '--warmup_epochs', '0',
        '--batch_size', '2', '--accum_iter', '2', '--lr', '.01', '--min_lr', '.01',
        '--synthetic_samples', '6'])
    args.optimizer_updates = 0
    log = RunLog(args.output_dir, args)
    print('Synthetic training checks:', log.root)
    try:
        model = ToyLoss()
        expected = copy.deepcopy(model)
        samples = torch.arange(1, 7).float().unsqueeze(1) / 6
        loader = DataLoader(TensorDataset(samples, torch.zeros(6)), batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        reference = torch.optim.SGD(expected.parameters(), lr=.01)
        train_one_epoch(model, loader, optimizer, torch.device('cpu'), 0, None, log, args)
        for batch in (samples[:4], samples[4:]):
            reference.zero_grad()
            expected(batch, .75)[0].backward()
            reference.step()
        torch.testing.assert_close(model.weight, expected.weight)
        assert args.optimizer_updates == 2
        print('PASS: accumulated gradients match larger batches, including a partial final window')

        mae = small_model(beta=.5)
        images = torch.randn(3, 3, 32, 32)
        validation = DataLoader(TensorDataset(images, torch.zeros(3)), batch_size=2)
        state = {k: v.clone() for k, v in mae.state_dict().items()}
        rng = torch.get_rng_state().clone()
        first = evaluate(mae, validation, torch.device('cpu'), args, 0, log)
        second = evaluate(mae, validation, torch.device('cpu'), args, 0, log)
        assert first == second and mae.training and torch.equal(rng, torch.get_rng_state())
        assert all(torch.equal(v, state[k]) for k, v in mae.state_dict().items())
        print('PASS: repeatable validation preserves weights, RNG and training mode')

        args.output_dir = str(log.root / 'checkpoints')
        args.min_decay, args.beta = .7, .5
        optimizer = torch.optim.AdamW(mae.parameters(), lr=.01)
        scaler = misc.NativeScalerWithGradNormCount()
        misc.save_model(args, 0, mae, mae, optimizer, scaler, save_last=True)
        args.resume = str(log.root / 'checkpoints/checkpoint-last.pth')
        restored = small_model(beta=0)
        misc.load_model(args, restored, optimizer, scaler)
        assert args.start_epoch == 1 and args.optimizer_updates == 2
        assert restored.method_settings() == mae.method_settings()
        torch.testing.assert_close(mae.forward_feat(images)[0], restored.forward_feat(images)[0], rtol=0, atol=0)
        print('PASS: checkpoint loader restores settings, counters and outputs')
        last = log.root / 'checkpoints/checkpoint-last.pth'
        numbered = last.with_name('checkpoint-00000.pth')
        numbered.hardlink_to(last)
        assert numbered.stat().st_ino == last.stat().st_ino
        with torch.no_grad():
            mae.cls_token.add_(1)
        misc.save_model(args, 1, mae, mae, optimizer, scaler, save_last=True)
        assert numbered.stat().st_ino != last.stat().st_ino
        old = torch.load(numbered, weights_only=True)
        new = torch.load(last, weights_only=True)
        torch.testing.assert_close(old['model']['cls_token'], restored.cls_token)
        torch.testing.assert_close(new['model']['cls_token'], mae.cls_token)
        print('PASS: numbered checkpoint shares storage and survives replacement of last')
    except BaseException as error:
        log.close('failed', str(error))
        raise
    log.close('completed')
    records = [json.loads(line) for line in (log.root / 'logs/metrics.jsonl').read_text().splitlines()]
    assert len([r for r in records if r['event'] == 'train_update']) == 2
    assert len([r for r in records if r['event'] == 'validation']) == 2
    train_records = [json.loads(line) for line in (log.root / 'logs/train/mae.jsonl').read_text().splitlines()]
    val_records = [json.loads(line) for line in (log.root / 'logs/validation/mae.jsonl').read_text().splitlines()]
    assert len(train_records) == 2 and all(r['event'] == 'train_update' for r in train_records)
    assert len(val_records) == 2 and all(r['event'] == 'validation' for r in val_records)
    print('PASS: training and validation records are saved in separate folders')
    print('PASS: update and validation measurements saved to', log.root)
    args.optimizer_updates, args.loss_name, args.lr_schedule = 0, 'velocity', 'constant'
    model = ToyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    callback_weights = []
    log = RunLog(Path(__file__).resolve().parents[1] / 'runs', args, loss_name='velocity')
    loss_fn = lambda network, samples, labels: network(samples, .75)[0]
    train_one_epoch(model, loader, optimizer, torch.device('cpu'), 0, None, log, args,
                    loss_fn=loss_fn, on_update=lambda: callback_weights.append(model.weight.item()))
    evaluate(model, loader, torch.device('cpu'), args, 0, log, loss_fn=loss_fn)
    log.close('completed')
    assert len(callback_weights) == 2 and args.optimizer_updates == 2
    torch.testing.assert_close(model.weight, expected.weight)
    assert 'velocity_train_loss' in (log.root / 'logs/train/velocity.jsonl').read_text()
    assert 'velocity_val_loss' in (log.root / 'logs/validation/velocity.jsonl').read_text()
    print('PASS: conditional loss callback, actual-update hook, constant LR and separate velocity records')


if __name__ == '__main__':
    main()
