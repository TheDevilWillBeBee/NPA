import os.path as osp
import torch
from torch.utils.data import Dataset
import torchvision.transforms as v1
import torchvision
from tqdm import tqdm
import numpy as np
import os




def _constant_density_point_sampler(density, threshold, jitter_scale):
    import sys
    sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
    from commons import geometry

    gshape, gmin, gsize = density * torch.ones(2, dtype=int), torch.zeros(2), torch.ones(2)
    gx = geometry.grange(gshape, gmin, gsize)
    def func(img: torch.Tensor) -> torch.Tensor:
        x = gx + torch.rand_like(gx) * gsize / gshape * jitter_scale
        v = geometry.bilinear_sample(x.view(-1, 2), img[0].T.contiguous(), gmin, gsize)
        idx = v > threshold
        return torch.concat([x.view(-1, 2)[idx], v[idx].view(-1, 1)], dim=-1)
    return func

def _farthest_point_sampler(count):
    import fpsample
    def func(x: torch.Tensor) -> torch.Tensor:
        pos_np = x[:,:2].detach().cpu().numpy()
        idx_np = fpsample.bucket_fps_kdline_sampling(pos_np, count, h=3)
        return x[torch.from_numpy(idx_np).to(x.device).long()]
    return func

def get_dataset(rootdir, num_points, seed=None, device='cpu'):
    if seed is not None:
        np.random.seed(seed)

    dataset_path = osp.join(rootdir, f'point-MNIST-{num_points}')

    if not osp.exists(dataset_path):
        mnist_transform = v1.Compose([
            v1.RandomVerticalFlip(p=1),
            v1.ToTensor(),
            v1.Lambda(_constant_density_point_sampler(224, threshold=0.5, jitter_scale=1.0)),
            v1.Lambda(_farthest_point_sampler(num_points)),
        ])

        print("Generating and saving training dataset...")
        train_data = _process_mnist_split(rootdir, True, mnist_transform, "training")
        os.makedirs(dataset_path, exist_ok=True)
        torch.save(train_data, osp.join(dataset_path, 'train.pt'))

        print("Generating and saving test dataset...")
        test_data = _process_mnist_split(rootdir, False, mnist_transform, "test")
        torch.save(test_data, osp.join(dataset_path, 'test.pt'))
        print("Dataset generation complete!")
    else:
        print("Reusing saved dataset...")

    train_data = torch.load(osp.join(dataset_path, 'train.pt'))
    test_data = torch.load(osp.join(dataset_path, 'test.pt'))

    train_data = tuple(arr.to(device) for arr in train_data)
    test_data = tuple(arr.to(device) for arr in test_data)
    print("Dataset loaded!")
    return train_data, test_data

def _process_mnist_split(rootdir, train, transform, split_name):
    mnist_split = torchvision.datasets.MNIST(
        root=rootdir,
        train=train,
        transform=transform,
        download=True,
    )
    data_list = [data for data in tqdm(mnist_split, desc=f"Processing {split_name} split")]
    x = torch.stack([data[0] for data in data_list])
    y = torch.tensor([data[1] for data in data_list], dtype=torch.long)
    return x, y
