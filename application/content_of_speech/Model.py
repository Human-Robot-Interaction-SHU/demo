from transformers import (
    Pipeline,
    PreTrainedTokenizer,
    ModelCard,
    PreTrainedModel,
    TFPreTrainedModel,
    BertPreTrainedModel,
    BertModel
)
from transformers.pipelines import ArgumentHandler
from typing import Union, Optional, List, Dict, Any
import numpy as np
import torch.nn as nn


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()

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
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class MultiLabelPipeline(Pipeline):
    def __init__(
            self,
            model: Union[PreTrainedModel, TFPreTrainedModel],
            tokenizer: PreTrainedTokenizer,
            modelcard: Optional[ModelCard] = None,
            framework: Optional[str] = None,
            task: str = "",
            args_parser: ArgumentHandler = None,
            device: int = -1,
            binary_output: bool = False,
            threshold: float = 0.3
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
            task=task
        )
        self.threshold = threshold

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        return self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)

    def _forward(self, model_inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self.model(**model_inputs)

    def postprocess(self, model_outputs: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        if isinstance(model_outputs, tuple):
            model_outputs = model_outputs[0]

        logits = model_outputs.detach().numpy()
        scores = 1 / (1 + np.exp(-logits))  # Apply sigmoid

        results = []
        for item in scores:
            labels = []
            score_values = []
            for idx, s in enumerate(item):
                if s > self.threshold:
                    labels.append(self.model.config.id2label[idx])
                    score_values.append(s)
            results.append({"labels": labels, "scores": score_values})
        return results

