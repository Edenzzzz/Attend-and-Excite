from typing import List, Dict, Optional
import torch

import sys 
sys.path.append(".")
sys.path.append("..")

from pipeline_attend_and_excite import AttendAndExcitePipeline
from config import RunConfig
from run import run_on_prompt, get_indices_to_alter
from utils import vis_utils
from utils.ptp_utils import AttentionStore
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--prompt", type=str, default="A person wearing a shit")
parser.add_argument("--from_file", type=str, default=None)

NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
stable = AttendAndExcitePipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda") 
tokenizer = stable.tokenizer


