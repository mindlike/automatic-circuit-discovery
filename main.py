import torch
import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import (
    HookedTransformer,
)

from acdc.docstring.utils import get_all_docstring_things
from acdc.ioi.utils import get_all_ioi_things
from acdc.reasoning.utils import get_all_reasoning_things

from acdc.acdc_utils import (
    make_nd_dict,
    reset_network,
    shuffle_tensor,
    cleanup,
    ct,
    TorchIndex,
    Edge,
    EdgeType,
)  # these introduce several important classes !!!

from acdc.TLACDCExperiment import TLACDCExperiment

from acdc.acdc_utils import kl_divergence
from acdc.induction.utils import get_all_induction_things
import argparse

torch.autograd.set_grad_enabled(False)

### Parser arguments for command-line
"""
--task=induction --zero-ablation --threshold=0.71 --indices-mode=reverse --first-cache-cpu=False --second-cache-cpu=False --max-num-epochs=100 --device=cpu
"""

TASK = "reasoning"

THRESHOLD = 0.71  # only used if >= 0.0
ZERO_ABLATION = True
INDICES_MODE = "reverse"
NAMES_MODE = "normal"
DEVICE = "cpu"
METRIC = "kl_div"
RESET_NETWORK = 0
SINGLE_STEP = False

second_metric = None  # some tasks only have one metric

if TASK == "reasoning":
    num_examples = 5
    things = get_all_reasoning_things(
        num_examples=num_examples, device=DEVICE, metric_name=METRIC,
        file1="data/yesno_train_small.txt",
        file2="data/yesno_train_small_corrupt.txt",
    )
elif TASK == "ioi":
    num_examples = 5
    things = get_all_ioi_things(
        num_examples=num_examples, device=DEVICE, metric_name=METRIC
    )
elif TASK == "induction":
    num_examples = 4
    seq_len = 300
    things = get_all_induction_things(
        num_examples=num_examples, seq_len=seq_len, device=DEVICE, metric=METRIC
    )

validation_metric = things.validation_metric # metric we use (e.g KL divergence)
toks_int_values = things.validation_data # clean data x_i
toks_int_values_other = things.validation_patch_data # corrupted data x_i'
tl_model = things.tl_model # transformerlens model

if RESET_NETWORK:
    reset_network(TASK, DEVICE, tl_model)

tl_model.reset_hooks()

# Save some mem
gc.collect()
torch.cuda.empty_cache()

tl_model.reset_hooks()

exp = TLACDCExperiment(
    model=tl_model,
    threshold=THRESHOLD,
    zero_ablation=ZERO_ABLATION,
    abs_value_threshold=THRESHOLD,
    ds=toks_int_values,
    ref_ds=toks_int_values_other,
    metric=validation_metric,
    second_metric=second_metric,
    verbose=True,
    indices_mode=INDICES_MODE,
    names_mode=NAMES_MODE,
    hook_verbose=False,
    add_sender_hooks=True,
    #use_pos_embed=use_pos_embed,
    add_receiver_hooks=False,
    remove_redundant=False,
    #show_full_index=use_pos_embed,
)


for i in range(num_examples):
    exp.step(testing=False)

    print(i, "-" * 50)
    print(exp.count_no_edges())

    if i == 0:
        exp.save_edges("edges.pkl")

exp.save_edges("another_final_edges.pkl")

exp.save_subgraph(
    return_it=True,
)


"""
parser = argparse.ArgumentParser(description="Used to launch ACDC runs. Only task and threshold are required")

task_choices = ['ioi', 'docstring', 'induction', 'tracr-reverse', 'tracr-proportion', 'greaterthan', 'or_gate']
parser.add_argument('--task', type=str, required=True, choices=task_choices, help=f'Choose a task from the available options: {task_choices}')
parser.add_argument('--threshold', type=float, required=True, help='Value for THRESHOLD')
parser.add_argument('--first-cache-cpu', type=str, required=False, default="True", help='Value for FIRST_CACHE_CPU (the old name for the `online_cache`)')
parser.add_argument('--second-cache-cpu', type=str, required=False, default="True", help='Value for SECOND_CACHE_CPU (the old name for the `corrupted_cache`)')
parser.add_argument('--zero-ablation', action='store_true', help='Use zero ablation')
parser.add_argument('--using-wandb', action='store_true', help='Use wandb')
parser.add_argument('--wandb-entity-name', type=str, required=False, default="remix_school-of-rock", help='Value for WANDB_ENTITY_NAME')
parser.add_argument('--wandb-group-name', type=str, required=False, default="default", help='Value for WANDB_GROUP_NAME')
parser.add_argument('--wandb-project-name', type=str, required=False, default="acdc", help='Value for WANDB_PROJECT_NAME')
parser.add_argument('--wandb-run-name', type=str, required=False, default=None, help='Value for WANDB_RUN_NAME')
parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
parser.add_argument("--wandb-mode", type=str, default="online")
parser.add_argument('--indices-mode', type=str, default="normal")
parser.add_argument('--names-mode', type=str, default="normal")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--reset-network', type=int, default=0, help="Whether to reset the network we're operating on before running interp on it")
parser.add_argument('--metric', type=str, default="kl_div", help="Which metric to use for the experiment")
parser.add_argument('--torch-num-threads', type=int, default=0, help="How many threads to use for torch (0=all)")
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument("--max-num-epochs",type=int, default=100_000)
parser.add_argument('--single-step', action='store_true', help='Use single step, mostly for testing')
parser.add_argument("--abs-value-threshold", action='store_true', help='Use the absolute value of the result to check threshold')
"""