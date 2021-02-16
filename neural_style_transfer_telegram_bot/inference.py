import argparse
import json
import os
from enum import Enum, auto
from tempfile import TemporaryDirectory

from . import cycle_gan_model, style_transfer_model
from .utils import save_images


class TaskType(Enum):
    photo2van_gogh = auto()
    photo2monet = auto()
    photo2cezanne = auto()
    photo2ukiyoe = auto()
    style_transfer = auto()


class Task:
    def __init__(self, type_):
        self._is_done = False
        self.type_ = type_
        self._dataroot = TemporaryDirectory()
        self._results_dir = TemporaryDirectory()

    @property
    def dataroot(self):
        if self._is_done:
            raise RuntimeError("task is done")
        return self._dataroot.name

    @property
    def results_dir(self):
        if self._is_done:
            raise RuntimeError("task is done")
        return self._results_dir.name

    def done(self):
        if not self._is_done:
            self._dataroot.cleanup()
            self._results_dir.cleanup()
            self._is_done = True


PRETRAINED_MODELS_DIR = os.path.join(
    os.path.dirname(__file__),
    "pretrained_models"
)

pretrained_model_names = {
    TaskType.photo2van_gogh: 'style_vangogh_pretrained',
    TaskType.photo2monet: 'style_monet_pretrained',
    TaskType.photo2cezanne: 'style_cezanne_pretrained',
    TaskType.photo2ukiyoe: 'style_ukiyoe_pretrained',
    TaskType.style_transfer: 'ImageNet_VGG_E_features_pretrained'
}
cycle_gan_task_types = (TaskType.photo2van_gogh, TaskType.photo2monet,
                        TaskType.photo2cezanne, TaskType.photo2ukiyoe)


def create_model_factory_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--low_resource_mode", action='store_true',
        help="whether to working with models using low resource configs"
    )
    options, _ = parser.parse_known_args()
    return options


class ModelNOptionsFactory:
    def __init__(self, options):
        self._created_models_n_opts = {}
        self.options = options

    def _create_model(self, task_type):
        name = pretrained_model_names[task_type]
        if self.options.low_resource_mode:
            config_file_name = "low_resource_model_configuration.json"
        else:
            config_file_name = "model_configuration.json"
        config_path = os.path.join(
            PRETRAINED_MODELS_DIR,
            name,
            config_file_name
        )
        with open(config_path) as config_file:
            config = json.load(config_file)
        config.extend([
            '--checkpoints_dir', PRETRAINED_MODELS_DIR,
            '--name', name])
        if task_type in cycle_gan_task_types:
            options = cycle_gan_model.CycleGANInferenceOptions(config).parse()
            model_cls = cycle_gan_model.CycleGANInferenceModel
        elif task_type is TaskType.style_transfer:
            options = style_transfer_model.create_options(config)
            model_cls = style_transfer_model.StyleTransferModel
        else:
            raise ValueError(f"unknown task_type `{task_type}`")
        model = model_cls(options)
        model.setup(options)
        if (
            task_type is not TaskType.style_transfer and
            options.eval
        ):
            model.eval()
        self._created_models_n_opts[task_type] = (model, options)
        return model, options

    def __call__(self, task_type):
        if not isinstance(task_type, TaskType):
            raise TypeError(f"task type muste be an instance of `{TaskType}`")
        model_n_opts = self._created_models_n_opts.get(task_type)
        if model_n_opts is None:
            model_n_opts = self._create_model(task_type)
        if self.options.low_resource_mode:
            del self._created_models_n_opts[task_type]
        return model_n_opts


get_model_n_options = ModelNOptionsFactory(create_model_factory_options())


def create_dataloader(task_type, options):
    if task_type is TaskType.style_transfer:
        return style_transfer_model.create_dataloader(options)
    elif task_type in cycle_gan_task_types:
        return cycle_gan_model.create_dataloader(options)
    else:
        raise ValueError(f"unknown task_type `{task_type}`")


class ImageProcessingError(Exception):
    pass


class ImageTooBigError(ImageProcessingError):
    pass


class ImageTooSmallError(ImageProcessingError):
    pass


def make_inference(task):
    model, opt = get_model_n_options(task.type_)
    opt.dataroot = task.dataroot
    opt.results_dir = task.results_dir
    dataloader = create_dataloader(task.type_, opt)
    try:
        for data in dataloader:
            model.set_input(data)
            model.produce_results()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            save_images(opt.results_dir, visuals, img_path,
                        aspect_ratio=opt.aspect_ratio)
    except RuntimeError as exc:
        if "CUDA out of memory" in exc.args[0]:
            raise ImageTooBigError(
                "not enough GPU memory to process image") from exc
        elif "DefaultCPUAllocator: not enough memory" in exc.args[0]:
            raise ImageTooBigError("not enough RAM to process image") from exc
        messages = ("Output size is too small",
                    "Padding size should be less than "
                    "the corresponding input dimension")
        if any(message in exc.args[0] for message in messages):
            raise ImageTooSmallError("not enough pixels to do processing "
                                     "by neural network") from exc
        else:
            raise
    except ValueError as exc:
        messages = ("height and width must be > 0",
                    "Expected more than 1 value per channel when training")
        if any(message in exc.args[0] for message in messages):
            raise ImageTooSmallError("not enough pixels to do processing "
                                     "by neural network") from exc
        else:
            raise
    model.cleanup()
