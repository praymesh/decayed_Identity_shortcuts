"""CPU metrics on complete patch grids and CLS features."""
import torch


def effective_rank(cls):
    # Paper section 6: A[i,j] is covariance between samples' CLS vectors.
    # Rows are samples, columns are channels; centre each row over channels.
    centered = cls.double() - cls.double().mean(dim=1, keepdim=True)
    covariance = centered @ centered.T / (cls.shape[1] - 1)
    values = torch.linalg.eigvalsh(covariance).clamp_min(0)
    if values.sum() == 0:
        return 0.0
    probability = values / values.sum()
    positive = probability[probability > 0]
    return -(positive * positive.log()).sum().item()  # entropy, without exp()


def spatial_energy(patches, grid_size, cutoff=.25):
    height, width = grid_size
    grid = patches.float().transpose(1, 2).reshape(-1, patches.shape[-1], height, width)
    centered = grid - grid.mean(dim=(-2, -1), keepdim=True)
    power = torch.fft.fft2(centered, norm='ortho').abs().square()
    fy, fx = torch.meshgrid(torch.fft.fftfreq(height), torch.fft.fftfreq(width), indexing='ij')
    high = fy.square() + fx.square() > cutoff**2
    energy = power.sum(dim=(1, 2, 3))
    high_energy = power[:, :, high].sum(dim=(1, 2))
    variation = .5 * ((grid[:, :, 1:] - grid[:, :, :-1]).square().mean(dim=(1, 2, 3)) +
                      (grid[:, :, :, 1:] - grid[:, :, :, :-1]).square().mean(dim=(1, 2, 3)))
    return energy, high_energy, variation


def knn_accuracy(train, labels, validation, targets, classes=100, k=20):
    train = torch.nn.functional.normalize(train.float(), dim=1)
    validation = torch.nn.functional.normalize(validation.float(), dim=1)
    correct = 0
    for start in range(0, len(validation), 64):
        similarities, indices = (validation[start:start + 64] @ train.T).topk(min(k, len(train)), dim=1)
        weights = ((similarities - similarities.max(dim=1, keepdim=True).values) / .07).exp()
        votes = torch.zeros(len(indices), classes)
        votes.scatter_add_(1, labels[indices], weights)
        correct += (votes.argmax(1) == targets[start:start + 64]).sum().item()
    return 100 * correct / len(validation)
