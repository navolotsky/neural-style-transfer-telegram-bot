"""Namespace containing packages from the official repository of
CycleGAN and pix2pix implementations on pytorch
(https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
"""

import os
import sys

cycle_gan_official_repository_gitcloned_root_dir = os.path.join(
    os.path.dirname(__file__),
    "pytorch-CycleGAN-and-pix2pix"
)

sys_path = sys.path
sys.path = [cycle_gan_official_repository_gitcloned_root_dir] + sys_path

import data
import models
import options
import util

sys.path = sys_path

del sys, os, sys_path
