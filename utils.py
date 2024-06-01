import torch
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image


def plot_with_masks(img: torch.Tensor, masks: torch.Tensor, num_labels: int) -> None:
    plt.figure()
    plt.imshow(to_pil_image(img))

    cm = plt.get_cmap('gist_rainbow')

    for label in range(1, num_labels):
        mask = (masks == label).float().squeeze()
        if mask.max() > 0:
            c = cm(label // 3 * 3.0 / num_labels)[:3]
            final_mask = torch.zeros(4, *mask.shape)
            final_mask[0, :, :] = mask * c[0]
            final_mask[1, :, :] = mask * c[1]
            final_mask[2, :, :] = mask * c[2]
            final_mask[3, :, :] = 0.5 * (mask > 0)

            plt.imshow(to_pil_image(final_mask))


    plt.show()
