import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    SiglipConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput


class CaptionQualityConfig(SiglipConfig):
    model_type = "caption_quality"

    def __init__(
        self,
        clip_model_name: str = "google/siglip-so400m-patch14-384",
        freeze_clip: bool = True,
        **kwargs,
    ) -> None:
        super(CaptionQualityConfig, self).__init__(**kwargs)
        self.clip_model_name = clip_model_name
        self.freeze_clip = freeze_clip


class CaptionQualityModel(PreTrainedModel):
    config_class = CaptionQualityConfig

    def __init__(
        self,
        config: AutoConfig,
    ) -> None:
        super(CaptionQualityModel, self).__init__(config)
        self.clip_model = AutoModel.from_pretrained(config.clip_model_name)

        if config.freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        num_features = (
            self.clip_model.config.vision_config.hidden_size
            + self.clip_model.config.text_config.hidden_size
        )

        self.accuracy = nn.Linear(num_features, 5)
        self.creativity = nn.Linear(num_features, 5)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        pixel_values = pixel_values.to(self.clip_model.device)
        input_ids = input_ids.to(self.clip_model.device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.clip_model.device)

        if labels is not None:
            labels = labels.to(self.clip_model.device)

        outputs = self.clip_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        image_features = outputs.image_embeds
        caption_features = outputs.text_embeds

        image_features = F.normalize(image_features, dim=-1)
        caption_features = F.normalize(caption_features, dim=-1)
        combined_features = torch.cat((image_features, caption_features), dim=-1)

        accuracy_logits = self.accuracy(combined_features)
        creativity_logits = self.creativity(combined_features)
        combined_logits = torch.cat((accuracy_logits, creativity_logits), dim=-1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            accuracy_loss = loss_fct(accuracy_logits, labels[:, 0])
            creativity_loss = loss_fct(creativity_logits, labels[:, 1])

            loss = (accuracy_loss + creativity_loss) / 2
            loss = loss.cpu()

        if not return_dict:
            output = (combined_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=combined_logits,
        )
