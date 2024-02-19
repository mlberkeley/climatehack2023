import numpy as np
import torch
import torch.nn.functional as F
import keys

normalizers = np.array([(-1.4087231368096667, 1.3890225067478146),
 (52.978670018652174, 1.4779275085314911),
 (31.545373449030897, 6.077633030692922),
 (3.017692725650799, 3.460296728623524),
 (178.186237936907, 47.040837795130464)])
device = "cuda" if torch.cuda.is_available() else "cpu"
means, stds = torch.tensor(
        normalizers[:, 0],
        dtype=torch.float, device = device
    ), torch.tensor(
        normalizers[:, 1],
        dtype=torch.float, device = device
    )

def site_normalize(vals):
    if keys.META.LATITUDE in vals:
        vals[keys.META.LATITUDE] = (vals[keys.META.LATITUDE] - means[0]) / stds[0]
    if keys.META.LONGITUDE in vals:
        vals[keys.META.LONGITUDE] = (vals[keys.META.LONGITUDE] - means[1]) / stds[1]
    if keys.META.ORIENTATION in vals:
        vals[keys.META.ORIENTATION] = (vals[keys.META.ORIENTATION] - means[2]) / stds[2]
    if keys.META.TILT in vals:
        vals[keys.META.TILT] = (vals[keys.META.TILT] - means[3]) / stds[3]
    if keys.META.KWP in vals:
        vals[keys.META.KWP] = (vals[keys.META.KWP] - means[4]) / stds[4]
    # return (vals - means) / stds
    return vals

#def denormalize(val_type, val):
    #assert val_type in norms, "val_type error, not in norms, can't be normalized"
    #mean, std = norms[val_type]
    #return (val * std) + mean



def shift_images(images, shift_x, shift_y):
    """
    Shift a batch of images in PyTorch tensor format.

    Args:
    - images (torch.Tensor): Batch of input images with shape (batch_size, channels, height, width)
    - shift_x (int): Number of pixels to shift in the horizontal direction
    - shift_y (int): Number of pixels to shift in the vertical direction

    Returns:
    - shifted_images (torch.Tensor): Batch of shifted images with the same shape as input images
    """

    batch_size, channels, height, width = images.size()

    # Create an output tensor to store shifted images
    shifted_images = torch.zeros_like(images)

    # Calculate the boundaries for cropping and filling
    crop_left = max(0, -shift_x)
    crop_right = max(0, shift_x)
    crop_top = max(0, -shift_y)
    crop_bottom = max(0, shift_y)

    fill_left = max(0, shift_x)
    fill_right = max(0, -shift_x)
    fill_top = max(0, shift_y)
    fill_bottom = max(0, -shift_y)

    # Shift images using torch.roll function
    shifted_images = torch.roll(images, shifts=(shift_y, shift_x), dims=(2, 3))

    # Crop and fill the shifted areas
    if crop_left > 0:
        shifted_images[:, :, :, :crop_left] = 0  # Fill left
    elif fill_left > 0:
        shifted_images[:, :, :, -fill_left:] = 0  # Fill left

    if crop_right > 0:
        shifted_images[:, :, :, -crop_right:] = 0  # Fill right
    elif fill_right > 0:
        shifted_images[:, :, :, :fill_right] = 0  # Fill right

    if crop_top > 0:
        shifted_images[:, :, :crop_top, :] = 0  # Fill top
    elif fill_top > 0:
        shifted_images[:, :, -fill_top:, :] = 0  # Fill top

    if crop_bottom > 0:
        shifted_images[:, :, -crop_bottom:, :] = 0  # Fill bottom
    elif fill_bottom > 0:
        shifted_images[:, :, :fill_bottom, :] = 0  # Fill bottom

    return shifted_images
