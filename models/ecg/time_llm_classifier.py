"""Time-LLM-inspired ECG classifier for MIT-BIH AAMI 5-class classification.

This module is intentionally separate from the existing BG forecasting models so
classification work does not break the original forecasting pipeline.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    DistilBertConfig,
    DistilBertModel,
    DistilBertTokenizer,
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer,
)
from transformers import logging as transformers_logging

from models.layers.Embed import PatchEmbedding
from models.layers.StandardNorm import Normalize

transformers_logging.set_verbosity_error()


class TimeLLMEcgClassifier(nn.Module):
    """Time-LLM-inspired classifier with a 5-way task head.

    Input:
        x: [batch, time, channels] where channels defaults to 1 for MIT-BIH.

    Output:
        logits: [batch, num_classes]
    """

    def __init__(self, configs: Dict):
        super().__init__()
        self.task_name = configs.get("task_name", "ecg_classification")
        self.sequence_length = int(configs.get("sequence_length", 256))
        self.enc_in = int(configs.get("enc_in", 1))
        self.d_model = int(configs.get("d_model", 64))
        self.d_ff = int(configs.get("d_ff", 128))
        self.dropout = float(configs.get("dropout", 0.1))
        self.patch_len = int(configs.get("patch_len", 16))
        self.stride = int(configs.get("stride", 8))
        self.num_classes = int(configs.get("num_classes", 5))
        self.pooling = configs.get("pooling", "mean")
        self.llm_model_name = configs.get("llm_model", "BERT")
        self.llm_dim = int(configs.get("llm_dim", 768))
        self.llm_layers = int(configs.get("llm_layers", 6))
        self.freeze_llm = bool(configs.get("freeze_llm", False))

        self.normalize = Normalize(self.enc_in, affine=False)
        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=self.dropout,
        )

        self.llm_model, self.tokenizer = self._build_llm_backbone()
        hidden_size = int(getattr(self.llm_model.config, "hidden_size", self.llm_dim))
        self.input_projection = nn.Linear(self.d_model, hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size, self.num_classes),
        )

        if self.freeze_llm:
            for param in self.llm_model.parameters():
                param.requires_grad = False

    def _build_llm_backbone(self):
        name = self.llm_model_name
        if name == "BERT":
            config = BertConfig.from_pretrained("bert-base-uncased")
            config.num_hidden_layers = min(self.llm_layers, config.num_hidden_layers)
            model = BertModel.from_pretrained("bert-base-uncased", config=config)
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif name == "DistilBERT":
            config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
            config.n_layers = min(self.llm_layers, config.n_layers)
            model = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        elif name == "GPT2":
            config = GPT2Config.from_pretrained("gpt2")
            config.n_layer = min(self.llm_layers, config.n_layer)
            model = GPT2Model.from_pretrained("gpt2", config=config)
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        elif name == "TinyBERT":
            model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
            tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        elif name == "MiniLM":
            model = AutoModel.from_pretrained("nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large")
            tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large")
        elif name == "MobileBERT":
            model = AutoModel.from_pretrained("google/mobilebert-uncased")
            tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        elif name == "ALBERT":
            model = AutoModel.from_pretrained("albert/albert-base-v2")
            tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
        elif name == "BERT-tiny":
            model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
            tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        elif name == "OPT-125M":
            model = AutoModel.from_pretrained("facebook/opt-125m")
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        elif name == "LLAMA":
            config = LlamaConfig.from_pretrained("huggyllama/llama-7b")
            config.num_hidden_layers = min(self.llm_layers, config.num_hidden_layers)
            model = LlamaModel.from_pretrained("huggyllama/llama-7b", config=config)
            tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError(f"Unsupported llm_model for ECG classification: {name}")
        return model, tokenizer

    def forward(self, x: torch.Tensor, metadata: Optional[Dict] = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, T, C], got {tuple(x.shape)}")

        x = self.normalize(x, "norm")
        x = x.permute(0, 2, 1)
        patches, n_vars = self.patch_embedding(x)
        patch_embeddings = self.input_projection(patches)
        backbone_outputs = self.llm_model(inputs_embeds=patch_embeddings)
        hidden = backbone_outputs.last_hidden_state
        pooled = self._pool_hidden(hidden)
        logits = self.classifier(pooled)
        return logits

    def _pool_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        if self.pooling == "last":
            return hidden[:, -1, :]
        if self.pooling == "center":
            center_idx = hidden.size(1) // 2
            return hidden[:, center_idx, :]
        raise ValueError(f"Unsupported pooling mode: {self.pooling}")
