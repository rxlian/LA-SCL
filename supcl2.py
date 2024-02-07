import argparse
import numpy as np
import pandas as pd
import re
import pickle
import json
from random import sample
from collections import Counter
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import random
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from transformers import (
    BertPreTrainedModel,
    BertTokenizer,
    BertModel,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    BertModel,
    BertForPreTraining
)

from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
import faiss
from utils import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import wandb

from model import BertWithLabel

logger = get_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=10, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Where to store the final model.")

    # dataset
    parser.add_argument("--dataset", type=str, default=None, 
                        help="huggingface dataset for testing.")
    parser.add_argument("--num_classes", type=int, default=None, 
                        help="Number of classes for classification.")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help=("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                            "sequences shorter will be padded if `--pad_to_max_lengh` is passed."))
    parser.add_argument("--pad_to_max_length", action="store_true",
                        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
    
    # model
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--config_name", type=str, default=None,
                        help="Pretrained config name or path if not the same as model_name.")

    # optimizer 
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate for supcl (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.1, 
                        help="Weight decay to use.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])

    # other settings
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform on labeled samples. If provided, overrides num_train_epochs.")
    parser.add_argument("--evaluation_steps", type=int, default=125,
                        help="evaluation_step")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_warmup_steps", type=int, default=0, 
                        help="Number of steps for the warmup in the lr scheduler.")
    
    parser.add_argument("--seed", type=int, default=42, 
                        help="A seed for reproducible training.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Where do you want to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--overwrite_cache", type=bool, default=False,
                        help="Overwrite the cached training and evaluation sets.")
    parser.add_argument("--checkpointing_steps", type=str, default=None,
                        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="If the training should continue from a checkpoint folder.")
    parser.add_argument("--with_tracking", action="store_true",
                        help="Whether to enable experiment trackers for logging.")
    parser.add_argument("--kmeans", action="store_true",
                        help="Whether to enable experiment trackers for logging.")
    parser.add_argument("--kmeans_only", action="store_true",
                        help="Whether to enable experiment trackers for logging.")
    parser.add_argument("--modified_kmeans", action="store_true",
                        help="Whether to enable experiment trackers for logging.")
    parser.add_argument("--ce_loss", action="store_true",
                        help="Whether to assign label similarity matrix.")
    parser.add_argument("--ce_loss_only", action="store_true",
                        help="Whether to assign label similarity matrix.")
    parser.add_argument("--cpcc", action="store_true",
                        help="Whether to use cpcc regularizer.")
    parser.add_argument("--tree_structure", action="store_true",
                        help="Whether to use tree_structure.")
    parser.add_argument("--update_label_emb", action="store_true",
                        help="Whether to update label embedding after several step of iterations.")
    parser.add_argument("--early_stop", action="store_true",
                        help="Whether to use early stop.")                        
    parser.add_argument("--method", type=str,
                        help="specify the method's name")
    parser.add_argument("--gpu_id", type=int, 
                        help="gpu id for kmeans")
    parser.add_argument("--valid_method", type=str,
                        help="valid method to save bestcheckpoints, either valid_loss or valid_acc")

    
    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    wandb.login()
    wandb.init(project="instance-centroid")

    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    accelerator = (
        Accelerator(project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # if args.seed is not None:
    set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    
    if args.dataset == 'wos':
        dataset = load_dataset("text", data_files={"train": "data/WebOfScience/WOS46985/X.txt"})
        label = load_dataset("text", data_files={"train": "data/WebOfScience/WOS46985/Y.txt"})
        dataset=dataset['train'].add_column('label', label['train']['text'])

        dataset = dataset.class_encode_column("label")

        train_test = dataset.train_test_split(test_size=0.1, stratify_by_column="label")
        train, test = train_test['train'], train_test['test']

        train_val = train.train_test_split(test_size=0.1, stratify_by_column='label')
        train, valid = train_val['train'], train_val['test']

    else:

        dataset = load_dataset(args.dataset, cache_dir="./data/", split='train')
        test = load_dataset(args.dataset, cache_dir="./data/", split="test")

        if args.dataset == 'DeveloperOats/DBPedia_Classes':
            l1, l2 = dataset['l1'], dataset['l2']
            l1_l2 = [str(l1[i])+','+str(l2[i]) for i in range(len(l1))]

            l1_test, l2_test = test['l1'], test['l2']
            l1_l2_test = [l1_test[i]+','+l2_test[i] for i in range(len(l1_test))]

            with open('data/label_str.pkl', 'rb') as f:
                mapping = pickle.load(f)

            labels = [mapping[i] for i in l1_l2]
            dataset = dataset.add_column("label", labels)

            labels_test = [mapping[i] for i in l1_l2_test]
            test = test.add_column("label", labels_test)

        dataset = dataset.class_encode_column("label")
        test = test.class_encode_column("label")

        train_val = dataset.train_test_split(test_size=0.01, stratify_by_column="label")
        train, valid = train_val['train'], train_val['test'].shuffle(seed=42)



    def get_few_shot(dataset):
        num = 50
        classes = dataset.unique('label')
        extracted_sentences, labels = [], []
        for class_name in classes:
            class_sentences = dataset.filter(lambda example: example['label'] == class_name)['text']
            total_len = len(class_sentences)
            if total_len > num:
                class_sentences = class_sentences[:num]
                total_len = num
            else:
                class_sentences = class_sentences

            extracted_sentences.extend(class_sentences)
            labels.extend(class_name for _ in range(total_len))
        new_dataset = Dataset.from_dict({'text': extracted_sentences})
        new_dataset = new_dataset.add_column('label', labels).shuffle(seed=42)
        return new_dataset


    train = get_few_shot(train)
    print(len(train))
    args.num_classes = len(Counter(list(train['label'])))
    print(args.num_classes)

    path = args.model_name_or_path

    tokenizer = BertTokenizer.from_pretrained(path)

    tree_matrix = None
    # calculate weights from tree structure
    if args.tree_structure:
        layer_weights, labels_str = get_tree_weights(dataset, args)
        layer_weights = layer_weights.to(device)
        tree_matrix = layer_weights

        
    label_matrix = None
    label_sim_matrix = None


    if args.method == 'label_string' or args.kmeans:
        label_encoder = BertModel.from_pretrained(path).to(device)

          
        if args.dataset == 'SetFit/20_newsgroups':
            _, labels_str = get_tree_weights(dataset, args)
            labels_str = [','.join(lb.split(',')) for lb in labels_str]

            labels_template = [f"It contains {label} news." for label in labels_str]
        
        elif args.dataset == 'wos':

            f = open('data/WebOfScience/Meta-data/label_mapping.json', 'r')
            labels_str = list(json.load(f).values())

            labels_template = []
            for label in labels_str:
                labels_template.append(f"It contains article in domain of {label}.")
            print(labels_template[0])

        elif args.dataset == 'DeveloperOats/DBPedia_Classes':
            labels_template = []

            for label in list(mapping.keys()):
                cat_lab = label.split(',')
                category, label = cat_lab[0], cat_lab[1]
                label = ' '.join(re.findall('[A-Z][^A-Z]*', label)[:])
                labels_template.append(f"It contains {label} under {category}.")
            # labels_template = [f"It contains {label}." for label in list(mapping.keys())]
        print(labels_template)

        tokenized_labels = tokenizer(labels_template, padding='longest', return_tensors='pt')

        label_encoder.eval()
        label_list = []
        
        label_matrix = extract_emb2(label_encoder, tokenizer, tokenized_labels, device)

        label_matrix_norm = label_matrix / label_matrix.norm(dim=1)[:,None]   
        label_sim_matrix = torch.mm(label_matrix_norm, label_matrix_norm.t())


    model = BertWithLabel.from_pretrained(path, num_labels=args.num_classes, tree_matrix=tree_matrix, \
        label_matrix=label_matrix, label_sim_mat=label_sim_matrix, tokenizer=tokenizer, args=args).to(device)


    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_length, padding="max_length" if args.pad_to_max_length else False, return_tensors='pt')

    tokenized_train = train.map(preprocess_function, batched=True)
    tokenized_valid = valid.map(preprocess_function, batched=True)
    tokenized_test = test.map(preprocess_function, batched=True)
    
    train_dataloader = DataLoader(tokenized_train, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size)
    valid_dataloader = DataLoader(tokenized_valid, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(tokenized_test, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
  
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("em_no_trainer", experiment_config)

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
    
    kmeans_res = None
    starting_epoch = 0
    completed_steps = 0
    lowest_valid_loss, highest_valid_acc, patience = 0, 0, 0
    break_flag1, break_flag2 = False, False
    for epoch in range(starting_epoch, args.num_train_epochs):

        model.train()
        if args.with_tracking:
            total_loss = 0

        logger.info(f"epoch = {epoch}")
        
        for step, batch in enumerate(train_dataloader):

            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue

            outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch["token_type_ids"].to(device),
                    labels=batch['labels'].to(device),
                    kmeans_res=kmeans_res,
                    args=args)
            
            wandb.log({"total loss": outputs.loss[0]})
            wandb.log({"instance-instance loss": outputs.loss[1]})
            wandb.log({"instance-centroid loss": outputs.loss[2]})

            loss = outputs.loss[0]

            if args.with_tracking:
                total_loss += loss.detach().float()

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if args.update_label_emb and (epoch*len(train_dataloader)+step) % 500 == 0 and step != 0:
                label_matrix = extract_emb2(model.bert, tokenizer, tokenized_labels, device)
     
                with torch.no_grad():
                    model.label_matrix.copy_(label_matrix)

                label_matrix_norm = label_matrix / label_matrix.norm(dim=1)[:,None]   
                label_sim_matrix = torch.mm(label_matrix_norm, label_matrix_norm.t())
                print(label_sim_matrix)

            if (step+1) % args.evaluation_steps == 0 or step == len(train_dataloader) - 1:
                logger.info(f"training loss: {total_loss.item()/(step+1)}")
                total_valid_loss = 0
                model.eval()
                labels, logits = [], []

                for step, batch in enumerate(valid_dataloader):
                    outputs = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        token_type_ids=batch["token_type_ids"].to(device),
                        labels=batch['labels'].to(device),
                        kmeans_res=kmeans_res,
                        args=args)
                    valid_loss = outputs.loss[0]
                    total_valid_loss += valid_loss.detach().float()

                    labels.append(batch['labels'].cpu().numpy())
                    logits.append(outputs.logits.argmax(-1).cpu().numpy())
                
                lb = np.concatenate(labels)
                pred = np.concatenate(logits)
                res = compute_metrics(lb, pred)
                acc = res['accuracy']

                logger.info(f"valid loss: {total_valid_loss.item()/len(valid_dataloader)}")
                logger.info(f"valid acc: {acc}")
                wandb.log({"valid acc": acc})

                if highest_valid_acc == 0 or acc > highest_valid_acc:
                    highest_valid_acc = acc

                    patience = 0
                    if args.output_dir is not None:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(args.output_dir)
                else:
                    patience += 1

                if args.early_stop:
                    if patience > 4:
                        break_flag1 = True
                        break
            if args.early_stop and (break_flag1 or completed_steps >= args.max_train_steps):
                break_flag2 = True
                break
        if args.early_stop and break_flag2:
            break
    

if __name__ == "__main__":
    main()
