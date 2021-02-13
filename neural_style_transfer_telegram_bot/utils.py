import os
from typing import Sequence

import torch
from PIL import Image
from torchvision.transforms import functional as F


def scale_by_biggest(img, size: int, interpolation=Image.BILINEAR):
    """Scale the input image to match its biggest edge to the given size.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): Image to be rescaled.
        size (int): Desired output size of the biggest edge of the image.
            The biggest edge will be matched to this. The smallest edge of
            the image will be rescaled maintaining the aspect ratio.
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.

    Returns:
            PIL Image or Tensor: Resized image.
    """
    if not isinstance(size, int):
        raise ValueError("Size must be an int, not a {}".format(type(size)))
    w, h = F._get_image_size(img)
    if (w <= h and h == size) or (h <= w and w == size):
        return img
    if w > h:
        new_w = size
        new_h = int(size * h / w)
    else:
        new_h = size
        new_w = int(size * w / h)
    return F.resize(img, (new_h, new_w), interpolation)


def limit_biggest_by_scaling(img, size: int, interpolation=Image.BILINEAR):
    """Scale the input image to match its biggest edge to the given size
    if the biggest edge is greater than the given size.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): Image to be rescaled.
        size (int): Desired output size of the biggest edge of the image.
            The biggest edge will be matched to this if it's size > the given size,
            else it is returned unchanged.
            The smallest edge of the image will be rescaled maintaining the aspect ratio
            if the biggest one is changed.
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.

    Returns:
            PIL Image or Tensor: Resized image.
    """
    if not isinstance(size, int):
        raise ValueError("Size must be an int, not a {}".format(type(size)))
    if max(F._get_image_size(img)) > size:
        return scale_by_biggest(img, size, interpolation)
    else:
        return img


def scale_then_crop_to_fit(img, size, interpolation=Image.BILINEAR):
    """Scale then crop the input image to fit to the given size.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): Image to be fitted.
        size (int): Desired output size in the following order: (h, w).
            The input image is rescaled to fit into the rectangle (h, w),
            then cropped at the center if some edge is out of the rectangle (h, w).
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.

    Returns:
            PIL Image or Tensor: Resized image.
    """
    if not isinstance(size, (tuple, list)):
        raise TypeError("Got inappropriate size arg")
    if len(size) != 2:
        raise ValueError("Size must be a 2 element tuple/list, not a "
                         "{} element tuple/list".format(len(size)))
    h, w = size  # Convention (h, w)
    image_w, image_h = F._get_image_size(img)
    if h == image_h and w == image_w:
        return img
    if not (
        (h == image_h and w < image_w) or
        (h < image_h and w == image_w)
    ):
        new_image_h = h
        new_image_w = int(h * image_w / image_h)
        if h > new_image_h or w > new_image_w:
            new_image_w = w
            new_image_h = int(w * image_h / image_w)
        img = F.resize(img, (new_image_h, new_image_w), interpolation)
    img = F.center_crop(img, (h, w))
    assert (w, h) == tuple(F._get_image_size(img)), (
        f"failed to fit the image (size: {image_h =}, {image_w =}) "
        f"to the given size: {h =}, {w =}")
    return img


class ScaleByBiggest(torch.nn.Module):
    """Scale the input image to match its biggest edge to the given size.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        size (int): Desired output size of the biggest edge of the image.
            The biggest edge will be matched to this. The smallest edge of
            the image will be rescaled maintaining the aspect ratio.
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size: int, interpolation=Image.BILINEAR):
        super().__init__()
        if not isinstance(size, int):
            raise TypeError("Size should be int. Got {}".format(type(size)))
        self.size = size
        self.interpolation = interpolation

    def forward(self, img):
        return scale_by_biggest(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = F._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class LimitBiggestByScaling(torch.nn.Module):
    """Scale the input image to match its biggest edge to the given size
    if the biggest edge is greater than the given size.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        size (int): Desired output size of the biggest edge of the image.
            The biggest edge will be matched to this if it's size > the given size,
            else it is returned unchanged.
            The smallest edge of the image will be rescaled maintaining the aspect ratio
            if the biggest one is changed.
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size: int, interpolation=Image.BILINEAR):
        super().__init__()
        if not isinstance(size, int):
            raise TypeError("Size should be int. Got {}".format(type(size)))
        self.size = size
        self.interpolation = interpolation

    def forward(self, img):
        return limit_biggest_by_scaling(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = F._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class ScaleThenCropToFit(torch.nn.Module):
    """Scale then crop the input image to fit to the given size.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        size (int): Desired output size in the following order: (h, w).
            The input image is rescaled to fit into the rectangle (h, w),
            then cropped at the center if some edge is out of the rectangle (h, w).
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__()
        if not isinstance(size, Sequence):
            raise TypeError(
                "Size should be sequence. Got {}".format(type(size)))
        if len(size) == 2:
            raise ValueError("Size should 2 values")
        self.size = size
        self.interpolation = interpolation

    def forward(self, img):
        return scale_then_crop_to_fit(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = F._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


def save_image(
    image_pil, image_path,
    aspect_ratio=1.0, method=Image.BICUBIC
):
    w, h = image_pil.size
    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), method)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), method)
    image_pil.save(image_path)


def save_images(
    image_dir, visuals, image_paths,
    aspect_ratio=1.0, method=Image.BICUBIC, ext=None
):
    for label, imgs_batch in visuals.items():
        for img_tensor, img_path in zip(imgs_batch, image_paths):
            short_path = os.path.basename(img_path)
            name, source_ext = os.path.splitext(short_path)
            if ext is None:
                source_ext = source_ext.lower()
                ext = source_ext if source_ext in ('.jpg', '.png') else '.jpg'
            image_name = "{}_{}{}".format(name, label, ext)
            save_path = os.path.join(image_dir, image_name)
            img = F.to_pil_image(img_tensor)
            save_image(img, save_path,
                       aspect_ratio=aspect_ratio, method=method)


def gather_file_paths(dir_path, *, ext=None):
    images = []
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} is not a valid directory")
    for root, _, fnames in sorted(os.walk(dir_path)):
        for filename in fnames:
            if ext is None or filename.endswith(ext):
                path = os.path.join(root, filename)
                images.append(path)
    return images


IMG_EXTENSIONS = (
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
)


def gather_image_paths(dir_path):
    return gather_file_paths(dir_path, ext=IMG_EXTENSIONS)
