"""TIGER-style generative Transformer for (c1,c2,c3,c4) Semantic IDs.

TIGER (Rajput et al., NeurIPS 2023)와 같게 둔 것
  - T5 encoder-decoder. encoder 입력 = [user token] + history 기사 SID token을 시간순으로 펼친 것
  - SID token은 level마다 다른 token (level offset을 둔 하나의 vocabulary)
  - decoder는 BOS부터 다음 클릭 기사의 SID token을 autoregressive하게 생성
  - loss = target token cross-entropy (teacher forcing, token 평균)
  - 출력 softmax는 SID vocabulary 전체 (level별로 따로 자르지 않음)

우리 SID에 맞춘 것
  - level 4개: c1(category, 25) / c2(128) / c3(512) / c4(collision suffix, max c4 + 1)
    TIGER도 3개 codeword + collision 구분 token 1개를 쓴다
  - 후보 5개의 점수 = 생성 log-likelihood (TIGER는 beam search로 전체 corpus에서 생성.
    여기서는 기존 1pos4neg 평가를 위해 같은 likelihood로 후보 5개를 채점)
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor, nn
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack

NUM_LEVELS = 4
PAD_ID = 0


class TigerGenerative(nn.Module):

    def __init__(
        self,
        level_sizes: List[int],
        num_user_buckets: int = 2000,
        d_model: int = 128,
        num_heads: int = 6,
        d_kv: int = 64,
        d_ff: int = 1024,
        num_layers: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        assert len(level_sizes) == NUM_LEVELS
        self.level_sizes = list(level_sizes)
        self.num_user_buckets = num_user_buckets

        # token id: 0 = PAD, 1.. = SID token (level offset), 그 뒤 = user token
        offsets, start = [], 1
        for size in level_sizes:
            offsets.append(start)
            start += size
        self.register_buffer("level_offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.sid_vocab_start = 1
        self.sid_vocab_size = sum(level_sizes)
        self.user_offset = start
        self.vocab_size = start + num_user_buckets

        self.token_embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=PAD_ID)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))

        def config(is_decoder: bool) -> T5Config:
            return T5Config(
                vocab_size=1, d_model=d_model, d_kv=d_kv, d_ff=d_ff,
                num_layers=num_layers, num_decoder_layers=num_layers, num_heads=num_heads,
                dropout_rate=dropout_rate, is_decoder=is_decoder,
                is_encoder_decoder=False, use_cache=False, eos_token_id=None,
            )

        self.encoder = T5Stack(config(False))
        self.decoder = T5Stack(config(True))
        self.lm_head = nn.Linear(d_model, self.sid_vocab_size, bias=False)

        nn.init.normal_(self.token_embedding.weight, std=1.0)
        nn.init.normal_(self.bos, std=1.0)
        with torch.no_grad():
            self.token_embedding.weight[PAD_ID].zero_()

    # ------------------------------------------------------------ tokens
    def sid_tokens(self, sids: Tensor) -> Tensor:
        """[..., 4] level code -> [..., 4] vocabulary id"""
        return sids + self.level_offsets

    def encode(self, history_sids: Tensor, history_mask: Tensor, user_bucket: Tensor):
        """history_sids [B,H,4] (오래된 것 -> 최근), history_mask [B,H], user_bucket [B]"""
        B, H, _ = history_sids.shape
        tokens = self.sid_tokens(history_sids).reshape(B, H * NUM_LEVELS)
        mask = history_mask.unsqueeze(-1).expand(-1, -1, NUM_LEVELS).reshape(B, H * NUM_LEVELS)
        tokens = torch.where(mask, tokens, torch.full_like(tokens, PAD_ID))
        user = (user_bucket + self.user_offset).unsqueeze(1)
        tokens = torch.cat([user, tokens], dim=1)
        mask = torch.cat([torch.ones_like(user, dtype=torch.bool), mask], dim=1)
        out = self.encoder(inputs_embeds=self.token_embedding(tokens),
                           attention_mask=mask.long(), return_dict=True)
        return out.last_hidden_state, mask.long()

    def decode_logits(self, target_sids: Tensor, enc_hidden: Tensor, enc_mask: Tensor) -> Tensor:
        """teacher forcing. target_sids [N,4] -> logits [N,4,sid_vocab]
        decoder 입력 = [BOS, t1, t2, t3], 위치 k에서 t_{k+1} 예측"""
        N = target_sids.shape[0]
        prev = self.token_embedding(self.sid_tokens(target_sids)[:, :NUM_LEVELS - 1])
        dec_in = torch.cat([self.bos.expand(N, -1, -1), prev], dim=1)
        out = self.decoder(inputs_embeds=dec_in, encoder_hidden_states=enc_hidden,
                           encoder_attention_mask=enc_mask, return_dict=True)
        return self.lm_head(out.last_hidden_state)

    def target_vocab_index(self, sids: Tensor) -> Tensor:
        """lm_head 출력 index (SID vocabulary 내부 위치)"""
        return self.sid_tokens(sids) - self.sid_vocab_start

    # ------------------------------------------------------------ train
    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        enc, mask = self.encode(batch["history_sids"], batch["history_mask"], batch["user_bucket"])
        target = batch["target_sids"]
        logits = self.decode_logits(target, enc, mask)
        labels = self.target_vocab_index(target)
        ce = nn.functional.cross_entropy(
            logits.float().reshape(-1, self.sid_vocab_size), labels.reshape(-1), reduction="none"
        ).reshape(-1, NUM_LEVELS)
        return {
            "loss": ce.mean(),                       # T5: target token 평균
            "level_ce": ce.detach().mean(0),         # [4]
            "level_correct": (logits.argmax(-1) == labels).float().detach().sum(0),
        }

    # ------------------------------------------------------------ scoring
    @torch.no_grad()
    def candidate_level_log_probs(self, batch: Dict[str, Tensor]) -> Tensor:
        """후보 [B,C,4]의 level별 log P(t_k | H, t_<k) [B,C,4]와 그 token이 argmax인지 [B,C,4]"""
        enc, mask = self.encode(batch["history_sids"], batch["history_mask"], batch["user_bucket"])
        cand = batch["candidate_sids"]
        B, C, _ = cand.shape
        enc_r = enc.unsqueeze(1).expand(-1, C, -1, -1).reshape(B * C, enc.shape[1], enc.shape[2])
        mask_r = mask.unsqueeze(1).expand(-1, C, -1).reshape(B * C, mask.shape[1])
        flat = cand.reshape(B * C, NUM_LEVELS)
        log_probs = torch.log_softmax(self.decode_logits(flat, enc_r, mask_r).float(), dim=-1)
        index = self.target_vocab_index(flat)
        picked = log_probs.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        correct = log_probs.argmax(-1) == index          # teacher forcing에서 그 token이 argmax인지
        return picked.reshape(B, C, NUM_LEVELS), correct.reshape(B, C, NUM_LEVELS)
