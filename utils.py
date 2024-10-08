import math

import torch
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image


def plot_with_masks(imgs: torch.Tensor, masks: torch.Tensor, num_labels: int) -> None:
    rows = math.floor(math.sqrt(imgs.shape[0]))
    cols = math.ceil(2*imgs.shape[0]/rows)

    fig, axes = plt.subplots(rows, cols)
    fig.set_figwidth(32)
    fig.set_figheight(16)

    for img_idx in range(imgs.shape[0]):
        img_row = math.floor(img_idx/(cols/2))
        img_col = int(2*(img_idx - img_row*(cols/2)))

        ax_img = axes[img_row, img_col]
        ax_img.imshow(to_pil_image(imgs[img_idx, :, :, :]))

        ax_mask = axes[img_row, img_col+1]
        cm = plt.get_cmap('gist_rainbow')

        for label in range(1, num_labels):
            mask = (masks[img_idx, :, :, :] == label).float().squeeze()
            if mask.max() > 0:
                c = cm(label * 3.0 / num_labels)[:3]
                final_mask = torch.zeros(4, *mask.shape)
                final_mask[0, :, :] = mask * c[0]
                final_mask[1, :, :] = mask * c[1]
                final_mask[2, :, :] = mask * c[2]
                final_mask[3, :, :] = (mask > 0)

                ax_mask.imshow(to_pil_image(final_mask))

    plt.show()
