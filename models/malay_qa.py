
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn

import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoModel,
    BertModel,
    BertConfig,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,

)
from transformers.modeling_outputs import QuestionAnsweringModelOutput

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class Malay_QA(nn.Module):


    """
    First adopt Bert LM to encode the global representations (get the features before performeding the QA task).
    Then input the obtained global representations into Outlooker to further encode the local information.
    Finally, execute the QA task.
    """

    def __init__(self, model_name_or_path, config_name="", cache_dir="", embed_dims=768, num_layers=4, \
        downsamples=False, num_heads=1, out_kernel=[3,768], out_padding=1, out_stride=1, mlp_ratios=3, \
        qkv_bias=False, qk_scale=False, attn_drop_rate=0., norm_layer=nn.LayerNorm, outlook_attention=True):

        super().__init__()

        self.embed_dims = embed_dims
        
        self.config = BertConfig.from_pretrained('malay-huggingface/bert-base-bahasa-cased')

        self.bert = BertModel.from_pretrained('malay-huggingface/bert-base-bahasa-cased')

        self.qa_outputs = nn.Linear(in_features=self.embed_dims, out_features=2)

        self._keys_to_ignore_on_save = None

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            save_config (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to save the config of the model. Useful when in distributed training like TPUs and need
                to call this function on all processes. In this case, set :obj:`save_config=True` only on the main
                process to avoid race conditions.
            state_dict (nested dictionary of :obj:`torch.Tensor`):
                The state dictionary of the model to save. Will default to :obj:`self.state_dict()`, but can be used to
                only save parts of the model or if special precautions need to be taken when recovering the state
                dictionary of a model (like when using model parallelism).
            save_function (:obj:`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace :obj:`torch.save` by another method.
            push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                .. warning::

                    Using :obj:`push_to_hub=True` will synchronize the repository you are pushing to with
                    :obj:`save_directory`, which requires :obj:`save_directory` to be a local clone of the repo you are
                    pushing to if it's an existing folder. Pass along :obj:`temp_dir=True` to use a temporary directory
                    instead.

            kwargs:
                Additional key word arguments passed along to the
                :meth:`~transformers.file_utils.PushToHubMixin.push_to_hub` method.
        """

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save the config
        if save_config:
            model_to_save.config.save_pretrained(save_directory)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            state_dict = {k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save}

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(state_dict, output_model_file)

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)


    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        #outputs: last_hidden_state, pooler_output, (hidden_states), (attentions)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1) # 分离出的start_logits/end_logits形状为([config.hidden_size, 1])
        start_logits = start_logits.squeeze(-1) # 降维至形状为([config.hidden_size])
        end_logits = end_logits.squeeze(-1) # 降维至形状为([config.hidden_size])


        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        #outputs = (start_logits, end_logits,) + outputs[2:]
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            #outputs = (total_loss,) + outputs

        #return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
        )