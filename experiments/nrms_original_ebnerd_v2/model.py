"""
NRMS (Neural News Recommendation with Multi-Head Self-Attention) 구현
원 논문: Wu et al., 2019, "Neural News Recommendation with Multi-Head Self-Attention"

구조:
  News Encoder : 제목 단어들 -> word embedding -> multi-head self-attention -> additive attention pooling -> news vector
  User Encoder : history news vector들 -> multi-head self-attention -> additive attention pooling -> user vector
  Click Score  : dot(user_vector, candidate_news_vector)
"""
import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """여러 벡터를 하나로 요약(pooling)할 때 쓰는 attention.
    NRMS에서 self-attention 출력들을 news/user vector 하나로 합칠 때 사용."""

    def __init__(self, dim, hidden_dim=200):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, dim)
        attn = torch.tanh(self.proj(x))          # (batch, seq_len, hidden_dim)
        score = self.query(attn).squeeze(-1)      # (batch, seq_len)
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        weight = torch.softmax(score, dim=-1)     # (batch, seq_len)
        out = torch.bmm(weight.unsqueeze(1), x).squeeze(1)  # (batch, dim)
        return out


class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, num_heads=20, attn_hidden=200, dropout=0.2):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.additive_attn = AdditiveAttention(embed_dim, attn_hidden)

    def forward(self, title_ids):
        # title_ids: (batch, title_len)
        mask = (title_ids != 0)
        x = self.dropout(self.embedding(title_ids))                      # (batch, title_len, embed_dim)
        key_padding_mask = ~mask
        # 제목 전체가 pad인 경우(히스토리 패딩 슬롯) attention 연산이 NaN 나는 것 방지
        all_pad = key_padding_mask.all(dim=1)
        safe_kpm = key_padding_mask.clone()
        safe_kpm[all_pad] = False
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=safe_kpm)  # (batch, title_len, embed_dim)
        # additive attention에도 "전부 마스킹된" row를 그대로 넘기면 softmax(all -inf) = NaN이 되고,
        # 이 NaN은 이후 masked_fill로 출력값을 0으로 덮어써도 역전파 과정에서 임베딩 등 공유 파라미터로
        # 새어나간다 (softmax의 backward가 자기 자신의 forward 값에 의존하기 때문). 그래서 self_attn과
        # 마찬가지로 "안전한" mask를 만들어 넘기고, 최종 출력만 0으로 덮어쓴다.
        safe_mask = mask.float().clone()
        safe_mask[all_pad] = 1.0
        news_vec = self.additive_attn(attn_out, mask=safe_mask)
        news_vec = news_vec.masked_fill(all_pad.unsqueeze(-1), 0.0)
        return news_vec  # (batch, embed_dim)


class UserEncoder(nn.Module):
    def __init__(self, news_dim, num_heads=20, attn_hidden=200, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_attn = nn.MultiheadAttention(news_dim, num_heads, batch_first=True)
        self.additive_attn = AdditiveAttention(news_dim, attn_hidden)

    def forward(self, history_news_vecs, history_mask):
        # history_news_vecs: (batch, hist_len, news_dim), history_mask: (batch, hist_len) 1=유효, 0=패딩
        # 클릭 히스토리가 0개인 유저(all-pad row)가 섞여 있으면 softmax가 전부 -inf를 받아 NaN이 남 ->
        # 그런 row는 "전부 유효하다"고 속여서 계산한 뒤, 결과를 0벡터로 덮어써서 안전하게 처리한다.
        all_pad = (history_mask.sum(dim=1) == 0)
        safe_mask = history_mask.clone()
        safe_mask[all_pad] = 1.0

        x = self.dropout(history_news_vecs)
        key_padding_mask = ~(safe_mask.bool())
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        user_vec = self.additive_attn(attn_out, mask=safe_mask.float())
        user_vec = user_vec.masked_fill(all_pad.unsqueeze(-1), 0.0)
        return user_vec  # (batch, news_dim)


class E5Encoder(nn.Module):
    """원본 e5 임베딩(768) -> news vector. 압축 안 한 상한선 비교용."""

    def __init__(self, in_dim=768, embed_dim=300, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, feats):
        return self.proj(self.dropout(feats))


class SIDEncoder(nn.Module):
    """시맨틱 ID (c1, c2, c3) -> news vector. 코드마다 임베딩을 학습해서 더한다.
    코드 값은 +1 해서 쓰고 0은 pad(영벡터)로 둔다."""

    def __init__(self, c1_size=18, c2_size=1024, c3_size=1024, embed_dim=300, dropout=0.2):
        super().__init__()
        self.e1 = nn.Embedding(c1_size + 1, embed_dim, padding_idx=0)
        self.e2 = nn.Embedding(c2_size + 1, embed_dim, padding_idx=0)
        self.e3 = nn.Embedding(c3_size + 1, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        # 기본 초기화 N(0,1)은 3개를 더하면 내적(점수)이 너무 커져 초기 loss가 폭주하므로 작게 초기화
        for e in (self.e1, self.e2, self.e3):
            nn.init.normal_(e.weight, std=0.1)
            with torch.no_grad():
                e.weight[0].zero_()

    def forward(self, codes):
        # codes: (..., 3), 값은 이미 +1 되어 있고 pad 행은 0
        v = self.e1(codes[..., 0]) + self.e2(codes[..., 1]) + self.e3(codes[..., 2])
        return self.dropout(v)


SID_MODES = ("sid", "sid_shuffled", "sid_c1", "sid_c12")


class NRMS(nn.Module):
    """
    input_mode:
      title        : 제목 단어 + self-attention (기본 NRMS)
      e5           : 원본 e5 임베딩 768차원
      sid          : 시맨틱 ID (c1,c2,c3)
      sid_shuffled : 시맨틱 ID를 기사들 사이에 무작위로 섞은 대조군
      sid_c1       : c1(카테고리)만 사용   (ablation)
      sid_c12      : c1+c2만 사용        (ablation)
    tables: {"title": (N,L) LongTensor, "e5": (N,768) FloatTensor, "sid": (N,3) LongTensor}
            중 input_mode에 필요한 것만 주면 된다. 맨 끝에 pad 행(행 번호 N)을 자동으로 붙인다.
    """

    def __init__(self, input_mode, table, vocab_size=0, embed_dim=300, num_heads=20, attn_hidden=200, dropout=0.2):
        super().__init__()
        self.input_mode = input_mode
        table = table.clone()
        pad = torch.zeros((1,) + tuple(table.shape[1:]), dtype=table.dtype)
        if input_mode in SID_MODES:
            table = table + 1  # 0은 pad 예약
        self.register_buffer("table", torch.cat([table, pad], dim=0), persistent=False)

        if input_mode == "title":
            self.news_encoder = NewsEncoder(vocab_size, embed_dim, num_heads, attn_hidden, dropout)
        elif input_mode == "e5":
            self.news_encoder = E5Encoder(table.shape[1], embed_dim, dropout)
        elif input_mode in SID_MODES:
            self.news_encoder = SIDEncoder(int(table[:, 0].max()), int(table[:, 1].max()), int(table[:, 2].max()),
                                           embed_dim, dropout)
        else:
            raise ValueError(input_mode)
        self.user_encoder = UserEncoder(embed_dim, num_heads, attn_hidden, dropout)

    def encode_news(self, rows):
        """rows: (batch, n) 행 번호 -> (batch, n, embed_dim)"""
        feats = self.table[rows]  # title:(b,n,L) / e5:(b,n,768) / sid:(b,n,3)
        if self.input_mode == "title":
            b, n, L = feats.shape
            return self.news_encoder(feats.reshape(b * n, L)).view(b, n, -1)
        return self.news_encoder(feats)

    def forward(self, history_rows, history_mask, candidate_rows):
        history_vecs = self.encode_news(history_rows)
        user_vec = self.user_encoder(history_vecs, history_mask)
        cand_vecs = self.encode_news(candidate_rows)
        return torch.bmm(cand_vecs, user_vec.unsqueeze(-1)).squeeze(-1)  # (batch, num_cand)
