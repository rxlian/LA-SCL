import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import (
    BertPreTrainedModel,
    BertModel,

)
from transformers.modeling_outputs import SequenceClassifierOutput


class BertWithLabel(BertPreTrainedModel):
    def __init__(self, config, label_matrix, tree_matrix, label_sim_mat, tokenizer, args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=True)
        self.pad_token_id = tokenizer.pad_token_id
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels, bias=False)
        
        # register label embedding matrix parameters for backprop
        if label_matrix is not None:
            self.label_matrix = nn.Parameter(label_matrix, requires_grad=False)
            # self.register_parameter('label_matrix', label_matrix)
            # self.label_matrix = label_matrix
            
            # learnable label sim matrix
            sim_matrix = nn.Parameter(label_sim_mat, requires_grad=True)
            self.register_parameter('sim_matrix', sim_matrix)
            

        else:
            self.label_matrix = label_matrix
        
        if tree_matrix is not None:
            self.tree_matrix = tree_matrix

        self.init_weights()

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
        kmeans_res=None,
        args=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        logits = None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # pooled_output = outputs.pooler_output
        x = outputs.last_hidden_state
        pad_mask = (input_ids != self.pad_token_id).float()
        pad_mask = pad_mask.unsqueeze(-1).expand(x.size()).float()
        x = x * pad_mask

        mean_pooling = x[:,1:-1,:].sum(dim=1) / pad_mask[:,1:-1].sum(dim=1)

        norm_pooled = F.normalize(mean_pooling, dim=-1)

        # supcon
        cosine_score = torch.matmul(norm_pooled, norm_pooled.T)
        
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        neg_mask = mask.logical_not().to(cosine_score.device)

        mask = mask.to(cosine_score.device)

        # enumerator
        anchor_dot_contrast = torch.div(cosine_score,0.3)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if args.method == 'supcl':
            # denominator
            exp_logits = torch.exp(logits) * neg_mask
            
        # scaled-supcon
        elif args.method == 'label_string':
            # scaled denominator
            label_col = (self.sim_matrix.shape[0] * labels).unsqueeze(-1)
            label_row = labels.unsqueeze(-2)   
            label_idx = label_col + label_row  
            
            weights = torch.take(self.sim_matrix, label_idx.flatten()).view(label_idx.shape) 
             
            scaled_anchor_dot_contrast = torch.div(cosine_score, (0.3*weights))
            # for numerical stability
            scaled_logits_max, _ = torch.max(scaled_anchor_dot_contrast, dim=1, keepdim=True)
            scaled_logits = scaled_anchor_dot_contrast - scaled_logits_max.detach()

            exp_logits = torch.exp(scaled_logits) * neg_mask


        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask*log_prob).sum(1) / mask.sum(1)

        loss0 = (-1 * mean_log_prob_pos).mean()

        # instance-centroid loss
        loss1 = 0
        # kmeans
        if args.kmeans:
            idx = labels.detach().cpu().tolist()

            norm_pos_prototypes = F.normalize(self.label_matrix, dim=-1)
            cosine_score_ic = torch.matmul(norm_pooled, norm_pos_prototypes.T)

            mask_ic = torch.zeros(cosine_score_ic.shape)
            mask_ic[[i for i in range(mask_ic.shape[0])],labels] = 1
            neg_mask_ic = mask_ic.logical_not().to(cosine_score_ic.device)
            mask_ic = mask_ic.to(cosine_score_ic.device)

            # enumerator
            anchor_dot_contrast_ic = torch.div(cosine_score_ic,0.3)
            # for numerical stability
            logits_max_ic, _ = torch.max(anchor_dot_contrast_ic, dim=1, keepdim=True)
            logits_ic = anchor_dot_contrast_ic - logits_max_ic.detach()

            if args.modified_kmeans:
                weights_ic = self.sim_matrix[labels]

                scaled_anchor_dot_contrast_ic = torch.div(cosine_score_ic, (0.3*weights_ic))
                # for numerical stability
                scaled_logits_max_ic, _ = torch.max(scaled_anchor_dot_contrast_ic, dim=1, keepdim=True)
                scaled_logits_ic = scaled_anchor_dot_contrast_ic - scaled_logits_max_ic.detach()

                exp_logits_ic = torch.exp(scaled_logits_ic) * neg_mask_ic
            
            else:

                anchor_dot_contrast_ic = torch.div(anchor_dot_contrast_ic, 0.3)
                exp_logits_ic = torch.exp(logits_ic) * neg_mask_ic

            log_prob_ic = logits_ic - torch.log(exp_logits_ic.sum(1, keepdim=True))
            mean_log_prob_pos_ic = (mask_ic*log_prob_ic).sum(1) / mask_ic.sum(1)

            loss1 = -(1*mean_log_prob_pos_ic).mean()


        total_loss = loss0 + loss1

        if args.method == 'supcl':
            output_logits = self.classifier(mean_pooling)
        else:
            output_logits = torch.matmul(mean_pooling, self.label_matrix.T)

        return SequenceClassifierOutput(
            loss=[total_loss, loss0, loss1],
            logits=output_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )