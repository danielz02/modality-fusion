import rasterio
import argparse
import numpy as np
from tqdm import tqdm


def make_patches(x_path: str, y_path: str, window_size: int):
    margin = window_size // 2

    with rasterio.open(x_path) as x:
        arr = x.read(list(range(1, 49)) + [50])
    with rasterio.open(y_path) as y:
        labels = y.read(1)

    x_patches, y_patches = [], []
    for label in tqdm(np.unique(labels)):
        if label == 0:
            continue
        for x_idx, y_idx in zip(*np.where(labels == label)):
            patch = arr[:, (x_idx - margin):(x_idx + margin + 1), (y_idx - margin):(y_idx + margin + 1)]
            _, h, w = patch.shape
            if h != window_size or w != window_size:
                continue
            x_patches.append(patch.copy())
            y_patches.append(label)

    x_patches = np.stack(x_patches, axis=0)
    y_patches = np.array(y_patches)
    print(x_patches.shape)

    np.save(f"{x_path.replace('.tif', '')}.npy", x_patches)
    np.save(f"{y_path.replace('.tif', '')}.npy", y_patches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    # parser.add_argument("--test-image", type=str)
    parser.add_argument("--labels", type=str)
    # parser.add_argument("--test-labels", type=str)
    args = parser.parse_args()
    make_patches(args.image, args.labels, 11)
