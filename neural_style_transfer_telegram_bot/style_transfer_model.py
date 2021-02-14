import argparse
import copy
import os
from collections import OrderedDict

import torch
from PIL import Image
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg
from torchvision.transforms import functional as V

from .utils import (LimitBiggestByScaling, ScaleByBiggest, gather_image_paths,
                    scale_then_crop_to_fit)


def set_gpu_ids(gpu_ids: str):
    """Set GPU ids to CUDA and return list of set ids.

    Parameters:
        gpu_ids -- string separated by comma ("like 0,1,2") or string "-1"
    Return:
        list of set ids like [0, 1, 2] or []
    """
    str_ids = gpu_ids.split(",")
    gpu_ids = []
    for str_id in str_ids:
        id_ = int(str_id)
        if id_ >= 0:
            gpu_ids.append(id_)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
    return gpu_ids


def register_device(net, gpu_ids=None):
    """Register CPU/GPU device (with multi-GPU support)"""
    if gpu_ids is None:
        gpu_ids = []
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    return net


def create_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot", type=str, default=":runtime-defined:",
        help="""path to images (pairs
                (name1_content.jpg, name1_style.jpg),
                (name2_content.jpg, name2_style.jpg), ...)""")
    parser.add_argument(
        "--name", type=str,
        default="ImageNet_VGG_E_features_pretrained",
        help="""name of the feature extractor network.
                It decides where checkpoint is stored""")
    parser.add_argument(
        "--gpu_ids", type=str, default="-1",
        help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
    parser.add_argument(
        "--checkpoints_dir", type=str, default="./checkpoints",
        help="models are saved here")
    parser.add_argument(
        "--results_dir", type=str, default="./results",
        help="saves results here.")
    parser.add_argument(
        "--save_best", action='store_true',
        help="""save the result with min loss function value,
                otherwise the lastest""")
    parser.add_argument(
        '--aspect_ratio', type=float, default=1.0,
        help='aspect ratio of result images')
    parser.add_argument(
        "--n_epochs", type=int, default=500,
        help="""number of epochs to transfer style from source to target""")
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="how many pairs will be processed in a batch")
    parser.add_argument(
        "--num_threads", default=0, type=int,
        help="number of threads for loading data")
    parser.add_argument(
        "--direction", type=str, default="AtoB",
        help="""AtoB to use 1st image as content and 2nd as style,
                BtoA to use in reversed""")
    parser.add_argument(
        "--preprocess", type=str, default="none",
        help="""resize, scale or do nothing to images at load time [resize |
                scale_by_biggest | limit_biggest_by_scaling | none]""")
    parser.add_argument(
        "--load_size", type=int, default=500,
        help="""resize images to this size (image will be squared),
                scale to match the biggest size to this size""")
    parser.add_argument(
        "--way_to_fit", type=str, default="none",
        help="""how to fit style image to content image
                in the model [resize | scale_then_crop | none]""")
    parser.add_argument(
        "--style_weight", type=int, default=100000,
        help="weight of style loss")
    parser.add_argument(
        "--content_weight", type=int, default=1,
        help="weight of content loss")
    return parser.parse_args(args)


def get_transform(options, method=Image.BICUBIC):
    transform_list = []
    if options.preprocess == "none":
        pass
    elif options.preprocess == "resize":
        transform_list.append(transforms.Resize(
            (options.load_size, options.load_size), method))
    elif options.preprocess == "scale_by_biggest":
        transform_list.append(ScaleByBiggest(options.load_size, method))
    elif options.preprocess == "limit_biggest_by_scaling":
        transform_list.append(LimitBiggestByScaling(options.load_size, method))
    else:
        raise ValueError(
            f"unknown preprocess argument value: {options.preprocess}")
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


class StyleTransferDataset(Dataset):
    def __init__(self, options):
        image_paths = gather_image_paths(options.dataroot)
        if len(image_paths) % 2 != 0:
            raise ValueError(
                "Number of images must be even "
                "because we combine them into pairs!")
        image_paths = sorted(image_paths)
        step = 2
        self.image_paths = [tuple(image_paths[i:i+step])
                            for i in range(0, len(image_paths), step)]
        self.transform = get_transform(options)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        content_path, style_path = self.image_paths[index]
        style = Image.open(style_path)
        content = Image.open(content_path)
        if self.transform is not None:
            style = self.transform(style)
            content = self.transform(content)
        return {"content": content, "content_path": content_path,
                "style": style, "style_path": style_path}


def create_dataloader(options):
    return DataLoader(
        StyleTransferDataset(options),
        batch_size=options.batch_size,
        num_workers=int(options.num_threads))


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer('mean', mean.view(-1, 1, 1), persistent=False)
        self.register_buffer('std', std.view(-1, 1, 1), persistent=False)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
        # to initialize with something
        self.loss = torch.zeros_like(self.target)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class GramMatrix(nn.Module):
    """Calculate the Gramian matrix separately for each sample in the batch
    and normalize values by dividing by a number of elements in the sample.
    """

    def forward(self, input):
        batch_size, f_map_num, h, w = input.size()
        features = input.view(batch_size, f_map_num, -1)
        transposed = torch.transpose(features, -2, -1)
        gm = torch.matmul(features, transposed)
        return gm.div_(f_map_num * h * w)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.gram_matrix = GramMatrix()
        self.target = self.gram_matrix(target_feature).detach()
        # to initialize with something
        self.loss = torch.zeros_like(self.target)

    def forward(self, input):
        gm = self.gram_matrix(input)
        self.loss = F.mse_loss(gm, self.target)
        return input


class StyleTransferModel:
    """Implements the Neural-Style algorithm (https://arxiv.org/abs/1508.06576)
    developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.

    Some part of the algorithm core implementation is adopted from Alexis Jacq
    (https://github.com/alexis-jacq/Pytorch-Tutorials/blob/d464a69f2e0e892ceed2a1a2cada12a52c180bc5/Neural_Style.py)

    But some part was changed: for example, gram matrix is due to
    it is incorrectly calculated (not per the image in the batch,
    but per the whole batch).
    """

    def __init__(self, options):
        gpu_ids = set_gpu_ids(options.gpu_ids)
        options.gpu_ids = gpu_ids
        self.gpu_ids = gpu_ids
        self.options = options
        if self.gpu_ids:
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
        else:
            self.device = torch.device('cpu')

        self.save_dir = os.path.join(options.checkpoints_dir, options.name)

        self.model_names = ['feature_extractor']
        self.visual_names = ['result']
        # Feature extractor from VGG19 (VGG Configuration E):
        self.net_feature_extractor = (vgg.make_layers(
            vgg.cfgs['E'], batch_norm=False).to(self.device).eval())
        # Disable unnecessary computations:
        for param in self.net_feature_extractor.parameters():
            param.requires_grad_(False)

        self.content_loss_cls = ContentLoss
        self.style_loss_cls = StyleLoss
        self.optimizer_input_image_cls = optim.LBFGS

        content_layers_default = ['conv_4']
        style_layers_default = [
            'conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        self.content_layers = content_layers_default
        self.style_layers = style_layers_default

        # normalization module
        MEAN = (0.485, 0.456, 0.406)  # ImageNet mean
        STD = (0.229, 0.224, 0.225)  # ImageNet std
        self.normalization = Normalization(MEAN, STD).to(self.device)

        self.image_paths = []
        self.content = self.style = self.result = None
        self.current_stylizing = {}

    def load_networks(self):
        """Load all the network from the disk.
        """
        for name in self.model_names:
            load_filename = "net_{}.pth".format(name)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, 'net_' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)

    def setup(self, options):  # options arg is for compatibility
        self.load_networks()

    def _get_style_model_and_losses(self):
        if self.content is None or self.style is None:
            raise RuntimeError("intended to invoking by `set_input`")
        content_img = self.content
        style_img = self.style
        cnn = self.net_feature_extractor

        # just in order to have an iterable access
        # to or list of content/syle losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        if self.normalization is not None:
            model = nn.Sequential(self.normalization)
        else:
            model = nn.Sequential()

        i = 0  # increment every time we see a conv

        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely
                # with the ContentLoss and StyleLoss we insert below.
                # So we replace with out-of-place ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(
                    layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = self.content_loss_cls(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = self.style_loss_cls(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if (
                isinstance(model[i], self.content_loss_cls) or
                isinstance(model[i], self.style_loss_cls)
            ):
                break

        model = model[:(i + 1)]
        if isinstance(self.net_feature_extractor, torch.nn.DataParallel):
            model = register_device(model, self.gpu_ids)

        return model, style_losses, content_losses

    def _get_input_optimizer(self):
        if self.result is None:
            raise RuntimeError("intended to invoking by `set_input`")
        # this line to show that input is a parameter that requires a gradient
        optimizer = self.optimizer_input_image_cls(
            [self.result.requires_grad_()])
        return optimizer

    @staticmethod
    def _fit_style_img_size_to_content_img_size(
            content, style, *, way_to_fit="none", method=Image.BICUBIC):
        """Transform style image to fit content image size.

        Just resizes ("resize") style image to content image size;
        scales ("scale_then_crop") image first to match one of sides of
        content image, then crops style image to match a second side of
        content image; does nothing ("none").

        Parameters:
            content (torch.Tensor) — content image
            style (torch.Tensor) — style image
            way_to_fit (str) — "scale_then_crop", "resize", or "none"
        """
        if way_to_fit == "none":
            return style
        content_w, content_h = V._get_image_size(content)
        if way_to_fit == "resize":
            return V.resize(style, (content_h, content_w), method)
        elif way_to_fit == "scale_then_crop":
            return scale_then_crop_to_fit(
                style, (content_h, content_w), method)
        else:
            raise ValueError(
                'way_to_fit must be "scale_then_crop", "resize" or "none"')

    def set_input(self, input):
        a_to_b = self.options.direction == 'AtoB'
        self.content = input['content' if a_to_b else 'style'].to(self.device)
        style = input['style' if a_to_b else 'content'].to(self.device)
        self.style = self._fit_style_img_size_to_content_img_size(
            self.content, style, way_to_fit=self.options.way_to_fit)
        assert self.content.shape == self.style.shape, (
            "content image and style image must have the same size",
            f"chosen way to fit: {self.options.way_to_fit}")
        self.image_paths = input['content_path' if a_to_b else 'style_path']
        self.result = self.content.clone()
        model, s_losses, c_losses = self._get_style_model_and_losses()
        optimizer = self._get_input_optimizer()
        self.current_stylizing = dict(
            model=model,
            style_losses=s_losses, content_losses=c_losses,
            optimizer=optimizer)

    def _run_style_transfer(self):
        if self.current_stylizing is None or self.result is None:
            raise RuntimeError("intended to invoking by `forward`")
        epochs_num = self.options.n_epochs
        style_weight = self.options.style_weight
        content_weight = self.options.content_weight
        model = self.current_stylizing['model']
        content_losses = self.current_stylizing['content_losses']
        style_losses = self.current_stylizing['style_losses']
        optimizer = self.current_stylizing['optimizer']
        best_result = input_img = self.result
        current_loss = min_loss = float('inf')
        save_best = self.options.save_best
        print('Optimizing..')
        epoch_num = 0
        while epoch_num <= epochs_num:
            def closure():
                nonlocal current_loss, epoch_num
                # correct the values
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                current_loss = loss.item()
                loss.backward()

                epoch_num += 1
                if epoch_num % 50 == 0:
                    print("Epoch {}:".format(epoch_num))
                    print("Style Loss : {:4f} Content Loss: {:4f}\n".format(
                        style_score.item(), content_score.item()))
                return style_score + content_score
            optimizer.step(closure)
            if save_best and current_loss < min_loss:
                min_loss = current_loss
                best_result = input_img.detach().clone()
        if save_best:
            self.result = best_result

    def produce_results(self):
        self._run_style_transfer()

    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(
        self, output_correction=lambda tensor: tensor.clamp(0, 1)
    ):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                tensor = getattr(self, name)
                if output_correction is not None:
                    tensor = output_correction(tensor)
                visual_ret[name] = tensor
        return visual_ret

    def cleanup(self):
        self.image_paths.clear()
        self.current_stylizing.clear()
        self.content = self.style = self.result = None
