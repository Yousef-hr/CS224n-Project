# coding=utf-8
import math
import os
import sys
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import (
    is_flash_attn_2_available,
    logging,
)
from transformers.models.llama.modeling_llama import LlamaModel

from .configuration_autoencoder import AutoencoderConfig
from .configuration_calm import CALMConfig
from .modeling_autoencoder import Autoencoder
from .modeling_calm import CALM, CustomCausalLMOutput


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

logger = logging.get_logger(__name__)

# Make project-root imports available when launched from language-modelling/calm.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.architectures import MoEMLP  # noqa: E402
from utils.losses import SIGRegGaussianOnly  # noqa: E402


class JEPAPredictor(nn.Module):
    """Deterministic latent predictor with optional MoE."""

    def __init__(self, config: CALMConfig):
        super().__init__()
        self.sample_noise_std = float(getattr(config, "jepa_sample_noise_std", 0.05))
        self.head_type = getattr(config, "jepa_head_type", "mlp")

        if self.head_type == "moe":
            num_experts = int(getattr(config, "jepa_num_experts", 4))
            self.body = MoEMLP(
                embed_dim=config.hidden_size,
                hidden_dim=config.hidden_size,
                num_experts=num_experts,
            )
        else:
            self.body = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.SiLU(),
            )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=1e-6),
            nn.Linear(config.hidden_size, config.latent_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.body(hidden_states))

    def sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        prediction = self.forward(hidden_states)
        if self.sample_noise_std <= 0:
            return prediction
        noise = self.sample_noise_std * torch.randn_like(prediction)
        return prediction + noise


class JEPATransformer(CALM):
    """
    JEPA variant of CALM.

    Replaces energy-score latent distribution matching with a deterministic latent
    prediction objective: MSE + SIGReg regularization.
    """

    config_class = CALMConfig

    def __init__(self, config):
        super().__init__(config)
        self.ae_config = AutoencoderConfig.from_pretrained(config.ae_path)
        self.ae_model = Autoencoder.from_pretrained(
            config.ae_path,
            config=self.ae_config,
        )
        for param in self.ae_model.parameters():
            param.requires_grad = False
        self.ae_model.eval()

        self.transformer = LlamaModel(config)
        self.generative_head = JEPAPredictor(config)
        self.padding_idx = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.patch_size = config.patch_size

        self.embed_proj = nn.Sequential(
            nn.Linear(self.patch_size * config.hidden_size, 2 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=1e-6),
        )

        self.lambda_sigreg = float(getattr(config, "lambda_sigreg", 0.1))
        self.eval_noise_std = float(getattr(config, "jepa_eval_noise_std", 0.02))
        self.sigreg = SIGRegGaussianOnly(num_slices=int(getattr(config, "sigreg_num_slices", 128)))

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if labels is None:
            labels = input_ids

        batch_size, seq_length = input_ids.size()
        patch_size = self.patch_size
        latent_length = seq_length // patch_size

        labels = labels[:, patch_size:]
        mask = labels.ne(-100)
        labels = labels[mask].unsqueeze(0)

        latent_states = self.ae_model.encoder(input_ids=labels).squeeze(0)
        target_mean, _ = torch.chunk(latent_states, 2, dim=-1)

        inputs_embeds = self.transformer.embed_tokens(input_ids).reshape(batch_size, latent_length, -1)[:, :-1, :]
        inputs_embeds = self.embed_proj(inputs_embeds)

        outputs = self.transformer(inputs_embeds=inputs_embeds)
        hidden_states = outputs[0]
        patch_mask = mask.reshape(batch_size, latent_length - 1, patch_size)[:, :, 0]
        hidden_states = hidden_states[patch_mask]

        latent_predictions = self.generative_head(hidden_states)

        mse_loss = F.mse_loss(latent_predictions, target_mean)
        sigreg_loss = self.sigreg(latent_predictions)
        loss = (1.0 - self.lambda_sigreg) * mse_loss + self.lambda_sigreg * sigreg_loss

        if not self.training:
            latent_eval = torch.stack(
                [
                    latent_predictions + self.eval_noise_std * torch.randn_like(latent_predictions),
                    latent_predictions + self.eval_noise_std * torch.randn_like(latent_predictions),
                ],
                dim=0,
            )
            return self.eval_brier(latent_eval, input_ids[:, patch_size:], outputs, loss)

        return CustomCausalLMOutput(
            loss=loss,
        )
