from collections import OrderedDict
from dataclasses import dataclass
import warnings
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
from typing import List
import click
import IPython
from acdc.acdc_utils import MatchNLLMetric, frac_correct_metric, logit_diff_metric, kl_divergence, negative_log_probs
import torch
from acdc.docstring.utils import AllDataThings
from tqdm import tqdm
from transformer_lens.HookedTransformer import HookedTransformer
from dataclasses import dataclass
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Callable
)

def get_gpt2_small(device="cuda") -> HookedTransformer:
    """
    Load a pre_trained HookedTransformer model from the transformer_lens package.
    """
    tl_model = HookedTransformer.from_pretrained("gpt2")
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    return tl_model

def get_ioi_gpt2_small(device="cuda"):
    """For backwards compatibility"""
    return get_gpt2_small(device=device)

def load_prompts(filepath: str):
    """
    Read in all texts in a .txt file. Each line is a new prompt that should be stored in a tuple with (prompt, correct_answer, wrong_answer).
    """
    with open(filepath, 'r') as file:
        prompts = [tuple(line.strip().split('\t')) for line in file]
    
    return prompts


def get_all_reasoning_things(num_examples, 
                             device,
                             metric_name,
                             file1,
                             file2,
                             kl_return_one_element=True,
                            ):

    # # Load the pre-trained model
    # tl_model = get_gpt2_small(device=device)

    # # Load the prompts
    # clean_data = [d for d in load_prompts(file1) if len(d[0]) < 105]
    # assert len(clean_data) >= num_examples * 2

    # # limit #examples to to twice num_examples to have validation and test set
    # clean_data = clean_data[:num_examples*2]

    # # Load the corrupt data ### NEEDS TO BE A DIFFERENT FILE (?)
    # corrupt_data = [d for d in load_prompts(file2) if len(d[0]) < 105]
    # assert len(corrupt_data) >= num_examples * 2
    # corrupt_data = corrupt_data[:num_examples*2]

    # tl_model.tokenizer.padding_side = "left"

    # clean_ = tl_model.tokenizer(clean_data, padding='max_length', return_tensors="pt")
    # corrupt_ = tl_model.tokenizer(corrupt_data, padding='max_length', return_tensors="pt")
    # max_length = max(max(clean_["input_ids"].shape[1]), max(corrupt_["input_ids"].shape[1]))

    # # extract only the prompt (no answer)
    # clean_data = [d[0] for d in clean_data]
    # # tokenize questions with left padding -- the padding means we needn't worry about seq_len
    # clean_input = tl_model.tokenizer(clean_data, padding=True, max_length=max_length, return_tensors="pt")
    # # tokenize answers
    # clean_answers = tl_model.tokenizer([" " + d[1] for d in clean_data], return_tensors="pt")

    # corrupt_data = [d[0] for d in corrupt_data]
    # corrupt_input = tl_model.tokenizer(corrupt_data, padding=True, max_length=max_length, return_tensors="pt")
    # corrupt_answers = tl_model.tokenizer([" " + d[1] for d in corrupt_data], return_tensors="pt")

    # loading model
    tl_model = get_gpt2_small(device=device)

    # loading clean prompts creating two splits
    clean_data = [d for d in load_prompts("data/yesno_train.txt") if len(d[0]) < 105]
    assert len(clean_data) >= num_examples * 2
    clean_data = clean_data[:num_examples*2]

    # loading corrupt prompts creating two splits
    corrupt_data = [d for d in load_prompts("data/yesno_train.txt") if len(d[0]) < 105]
    assert len(corrupt_data) >= num_examples * 2
    corrupt_data = corrupt_data[:num_examples*2]

    # specifying padding side (left)
    tl_model.tokenizer.padding_side = "left"

    # extracting questions only
    clean_data = [d[0] for d in clean_data]
    corrupt_data = [d[0] for d in corrupt_data]

    # finding the longest tokenised sequence
    clean_ = tl_model.tokenizer(clean_data, padding='longest', return_tensors="pt")
    corrupt_ = tl_model.tokenizer(corrupt_data, padding='longest', return_tensors="pt")
    max_length = max(clean_["input_ids"].shape[1], corrupt_["input_ids"].shape[1])

    # tokeinising questions with left padding
    clean_input = tl_model.tokenizer(clean_data, padding='max_length', max_length=max_length, return_tensors="pt")
    clean_answers = tl_model.tokenizer([" " + d[1] for d in clean_data], return_tensors="pt")
    corrupt_input = tl_model.tokenizer(corrupt_data, padding='max_length', max_length=max_length, return_tensors="pt")
    corrupt_answers = tl_model.tokenizer([" " + d[1] for d in corrupt_data], return_tensors="pt")

    # tokenized objects contain inputs_ids and an attention mask
    default_data = clean_input["input_ids"].to(device)
    patch_data = corrupt_input["input_ids"].to(device)

    labels = clean_answers["input_ids"].to(device)
    corrupt_labels = corrupt_answers["input_ids"].to(device)

    validation_data = default_data[:num_examples]
    validation_patch_data = patch_data[:num_examples]
    validation_labels = labels[:num_examples]
    validation_corrupt_labels = corrupt_labels[:num_examples]

    test_data = default_data[num_examples:]
    test_patch_data = patch_data[num_examples:]
    test_labels = labels[num_examples:]
    test_wrong_labels = corrupt_labels[num_examples:]

    with torch.no_grad():
        #base_model_logits = tl_model(default_data)[:, -1, :]
        base_model_logits = tl_model(default_data)[:, -1, :]
        base_model_logprobs = F.log_softmax(base_model_logits, dim=-1)

    base_validation_logprobs = base_model_logprobs[:num_examples, :]
    base_test_logprobs = base_model_logprobs[num_examples:, :]


    if metric_name == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
            return_one_element=kl_return_one_element,
        )
    elif metric_name == "logit_diff":
        validation_metric = partial(
            logit_diff_metric,
            correct_labels=validation_labels,
            wrong_labels=validation_corrupt_labels,
        )
    elif metric_name == "frac_correct":
        validation_metric = partial(
            frac_correct_metric,
            correct_labels=validation_labels,
            wrong_labels=validation_corrupt_labels,
        )
    elif metric_name == "nll":
        validation_metric = partial(
            negative_log_probs,
            labels=validation_labels,
            last_seq_element_only=True,
        )
    elif metric_name == "match_nll":
        validation_metric = MatchNLLMetric(
            labels=validation_labels,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
        )
    else:
        raise ValueError(f"metric_name {metric_name} not recognized")

    test_metrics = {
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
        ),
        "logit_diff": partial(
            logit_diff_metric,
            correct_labels=test_labels,
            wrong_labels=test_wrong_labels,
        ),
        "frac_correct": partial(
            frac_correct_metric,
            correct_labels=test_labels,
            wrong_labels=test_wrong_labels,
        ),
        "nll": partial(
            negative_log_probs,
            labels=test_labels,
            last_seq_element_only=True,
        ),
        "match_nll": MatchNLLMetric(
            labels=test_labels,
            base_model_logprobs=base_test_logprobs,
            last_seq_element_only=True,
        ),
    }

    return AllDataThings(
        tl_model=tl_model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=validation_labels,
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=None,
        test_patch_data=test_patch_data,
    )


@dataclass(frozen=False)
class AllDataThings:
    tl_model: HookedTransformer
    validation_metric: Callable[[torch.Tensor], torch.Tensor]
    validation_data: torch.Tensor
    validation_labels: Optional[torch.Tensor]
    validation_mask: Optional[torch.Tensor]
    validation_patch_data: torch.Tensor
    test_metrics: dict[str, Any]
    test_data: torch.Tensor
    test_labels: Optional[torch.Tensor]
    test_mask: Optional[torch.Tensor]
    test_patch_data: torch.Tensor


@dataclass(frozen=True)
class Conn:
    inp: str
    out: str
    qkv: tuple[str, ...]



"""
    ioi_dataset = IOIDataset(
        prompt_type="ABBA",
        N=num_examples*2,
        nb_templates=1,
        seed = 0,
    )

    abc_dataset = (
        ioi_dataset.gen_flipped_prompts(("IO", "RAND"), seed=1)
        .gen_flipped_prompts(("S", "RAND"), seed=2)
        .gen_flipped_prompts(("S1", "RAND"), seed=3)
    )
"""