import logging
from collections import Counter
from dataclasses import dataclass, field
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import datasets
from datasets import load_dataset, concatenate_datasets
from datasets.arrow_dataset import Dataset
from transformers import (
        BertTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        TrainingArguments,
        Trainer,
        AutoModel,
        BertModel,
        AutoConfig,
        BertPreTrainedModel,
        add_start_docstrings,
        HfArgumentParser,
        set_seed,
        EarlyStoppingCallback
)

from transformers.file_utils import add_start_docstrings_to_model_forward, add_code_sample_docstrings
from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.trainer_utils import is_main_process
import wandb
from mytrainer import MyTrainer
from utils import *
import pickle
import re


logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The training data file (.txt or .csv)."}
    )

    valid_file: Optional[str] = field(
        default=None,
        metadata={"help": "The validation data file"}
    )

    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "The test data file"}
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset name"}
    )
    num_classes: Optional[int] = field(
        default=None,
        metadata={"help": "Num of classes"}
    )

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    task: Optional[str] = field(
        default='lp',
        metadata={
            "help": "either linear probe or test directly."
        }
    )
    

BERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.init_weights()
    """
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    """
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.bert.eval()

        for param in self.bert.parameters():
            param.requires_grad=False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        mean_pooling = outputs.last_hidden_state[:,1:-1,:].mean(dim=-2)
        logits = self.classifier(mean_pooling)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def compute_metrics_with_label_str(label_str):
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
      
        labels_str = np.asarray(label_str)[labels]
        preds_str = np.asarray(label_str)[preds]

        # acc of one layer above the leaf node
        labels_parent = [i.split(',')[-2] if i != 'neutral' else i.split(',')[-1] for i in labels_str]
        preds_parent = [i.split(',')[-2] if i != 'neutral' else i.split(',')[-1] for i in preds_str]
        parent_acc = sum(i==j for i,j in zip(labels_parent, preds_parent)) / len(labels)
        _, _, parent_f1, _ = precision_recall_fscore_support(labels_parent, preds_parent, average='macro')

        # root layer acc
        labels_root = [i.split(',')[0] for i in labels_str]
        preds_root = [i.split(',')[0] for i in preds_str]
        root_acc = sum(i==j for i,j in zip(labels_root, preds_root)) / len(labels)
        _, _, root_f1, _ = precision_recall_fscore_support(labels_root, preds_root, average='macro')

        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall, \
                "root_acc": round(root_acc, 4), "root_f1":round(root_f1,4), "parent_acc": round(parent_acc, 4), "parent_f1":round(parent_f1,4)}
    return compute_metrics


def compute_metrics_each_class(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        class_accuracies = []

        for class_ in np.unique(labels):
            class_acc = np.mean(preds[labels == class_] == class_)
            class_accuracies.append(class_acc)


        ly2 = [class_accuracies[i] for i in [0,1,3,4,5,6,16,17,18]]
        ly3 = [class_accuracies[i] for i in [2,7,8,9,10,11,15,19]]
        ly4 = [class_accuracies[i] for i in [14]]
        ly5 = [class_accuracies[i] for i in [12,13]]


        return {'mean':np.mean(class_accuracies), 'layer2':np.mean(ly2),'layer3':np.mean(ly3),\
                'layer4':np.mean(ly4),'layer5':np.mean(ly5)}

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

 
    train = load_dataset("csv", data_files=data_args.train_file, cache_dir="./data/")['train'].shuffle(seed=42)
    valid = load_dataset("csv", data_files=data_args.valid_file, cache_dir="./data/")['train'].shuffle(seed=42)
    test = load_dataset(data_args.dataset, cache_dir="./data/", split='test').shuffle(seed=42)

    if data_args.dataset == 'DeveloperOats/DBPedia_Classes':

        l1, l2 = test['l1'], test['l2']
        l1_l2 = [l1[i]+','+l2[i] for i in range(len(l1))]

        with open('label_str.pkl', 'rb') as f:
            mapping = pickle.load(f)

        labels = [mapping[i] for i in l1_l2]
        test = test.add_column("label", labels)


    path = model_args.model_name_or_path

    tokenizer = BertTokenizer.from_pretrained(path)
    data_args.num_classes = len(Counter(list(test['label'])))

    MODEL_PATH = "baseline_models/checkpoint_best_micro.pt"
    state_dict = torch.load(MODEL_PATH)["model"]
    config = AutoConfig.from_pretrained("./bert_config.json")
    model = BertModel(config)

    model = BertModel._load_state_dict_into_model(
        model,
        state_dict,
        MODEL_PATH
    )[0]

    # make sure token embedding weights are still tied if needed
    model.tie_weights()

    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()

    if "label_matrix" in state.keys():
        label_matrix = state['label_matrix']

    if data_args.dataset == 'SetFit/20_newsgroups':

        _, labels_str = get_tree_weights(test, data_args)
        labels_str = [','.join(lb.split(',')) for lb in labels_str]
        labels_template = [f"It contains {label} news." for label in labels_str]


    elif data_args.dataset == 'DeveloperOats/DBPedia_Classes':

        labels_str = list(mapping.keys())
        labels_template = []

        for label in labels_str:
            cat_lab = label.split(',')
            category, label = cat_lab[0], cat_lab[1]
            label = ' '.join(re.findall('[A-Z][^A-Z]*', label)[:])
            labels_template.append(f"It contains {label} under {category}.")

    else:

        f = open('data/WebOfScience/Meta-data/label_mapping.json', 'r')
        labels_str = list(json.load(f).values())

        labels_template = []
        for label in labels_str:
            labels_template.append(f"It contains domain of {label}.")
    
    if "label_matrix" not in state.keys():
        tokenized_labels = tokenizer(labels_template, padding='longest', return_tensors='pt')

        label_matrix = extract_emb2(model.bert, tokenizer, tokenized_labels, device)

    with torch.no_grad():
            model.classifier.weight.copy_(label_matrix)
            print("initialize linear layer with pretrained label embedding matrix")

    # random initialized linear-layer classifier
    if data_args.task == 'lp_random_initialized_linear':
        model.classifier.reset_parameters()
        print("randomly initialized classifier parameters!")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    def preprocess_function(examples):
        return tokenizer(str(examples["text"]), truncation=True, max_length=data_args.max_seq_length, padding=True)
    
    tokenized_imdb_train = train.map(preprocess_function)
    tokenized_imdb_valid = valid.map(preprocess_function)
    tokenized_imdb_test = test.map(preprocess_function)

    trainer = MyTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics_with_label_str(label_str=labels_str),
        train_dataset=tokenized_imdb_train,
        eval_dataset=tokenized_imdb_valid,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    if data_args.task == 'lp' or data_args.task == 'lp_random_initialized_linear' or data_args.task == 'finetune':
        trainer.train()
        trainer.save_model()
    
    trainer.evaluate(eval_dataset=tokenized_imdb_test)    

if __name__ == '__main__':
    main()



