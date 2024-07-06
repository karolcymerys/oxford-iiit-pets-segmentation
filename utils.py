import math

import torch
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image


def plot_with_masks(imgs: torch.Tensor, masks: torch.Tensor, num_labels: int) -> None:
    rows = math.floor(math.sqrt(imgs.shape[0]))
    cols = math.ceil(imgs.shape[0]/rows)

    fig, axes = plt.subplots(rows, cols)
    fig.set_figwidth(16)
    fig.set_figheight(16)

    for img_idx in range(imgs.shape[0]):
        img_row = math.floor(img_idx/cols)
        img_col = img_idx - img_row*cols
        ax = axes[img_row, img_col]
        ax.imshow(to_pil_image(imgs[img_idx, :, :, :]))

        cm = plt.get_cmap('gist_rainbow')

        for label in range(1, num_labels):
            mask = (masks[img_idx, :, :, :] == label).float().squeeze()
            if mask.max() > 0:
                c = cm(label // 3 * 3.0 / num_labels)[:3]
                final_mask = torch.zeros(4, *mask.shape)
                final_mask[0, :, :] = mask * c[0]
                final_mask[1, :, :] = mask * c[1]
                final_mask[2, :, :] = mask * c[2]
                final_mask[3, :, :] = 0.5 * (mask > 0)

                ax.imshow(to_pil_image(final_mask))

    plt.show()
