import argparse
from collections import OrderedDict
from itertools import chain

from .cycle_gan_and_pix2pix_official import data, models, options
from data import create_dataset
from models.test_model import TestModel
from options.test_options import TestOptions


def create_dataloader(options):
    return create_dataset(options)


class InferenceOptions(TestOptions):
    _default_args = [
        ("--serial_batches",),
        ("--no_flip",),
        ("--dataroot", ":runtime-defined:"),
        ("--num_threads", "0"),
        ("--batch_size", "1"),
        ("--gpu_ids", "-1")
    ]

    def __init__(self, pseudo_commandline_args):
        super().__init__()
        for argval in self._default_args:
            arg = argval[0]
            if arg not in pseudo_commandline_args:
                pseudo_commandline_args.extend(argval)

        self._orig_arg_parser = orig_arg_parser = argparse.ArgumentParser

        # Some monkey patching to not modify
        # the original pytorch-CycleGAN-and-pix2pix code:

        class FakeArgumentParser(argparse.ArgumentParser):
            def __init__(self, *args, **kwargs):
                argparse.ArgumentParser = orig_arg_parser
                super().__init__(*args, **kwargs)
                argparse.ArgumentParser = self.__class__

            def parse_known_args(self, args=None, namespace=None):
                if args is None:
                    args = pseudo_commandline_args
                return super().parse_known_args(args=args, namespace=namespace)

            def parse_args(self, args=None, namespace=None):
                if args is None:
                    args = pseudo_commandline_args
                return super().parse_args(args=args, namespace=namespace)

        self._fake_arg_parser = FakeArgumentParser

    def gather_options(self):
        argparse.ArgumentParser = self._fake_arg_parser
        result = super().gather_options()
        argparse.ArgumentParser = self._orig_arg_parser
        return result

    def print_options(self, opt):
        pass


class CycleGANInferenceOptions(InferenceOptions):
    _default_args = [
        ("--model", "test"),
        ("--preprocess", "none"),
        ("--no_dropout",),
        ("--eval",),
    ]
    for argval in InferenceOptions._default_args:
        arg = argval[0]
        if arg not in chain(_default_args):
            _default_args.append(argval)


class CycleGANInferenceModel(TestModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.visual_names = ['result']

    @property
    def result(self):
        return self.fake

    def produce_results(self):
        self.test()

    def cleanup(self):
        self.image_paths.clear()
        self.real = self.fake = None

    def get_current_visuals(
            self, output_correction=lambda tensor: (tensor + 1) / 2):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                tensor = getattr(self, name)
                if output_correction is not None:
                    tensor = output_correction(tensor)
                visual_ret[name] = tensor
        return visual_ret
