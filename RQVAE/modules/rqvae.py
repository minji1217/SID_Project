import torch

from functools import cached_property
from huggingface_hub import PyTorchModelHubMixin
from typing import List, NamedTuple, Optional

from torch import nn
from torch import Tensor

from modules.encoder import MLP
from modules.loss import ReconstructionLoss, RqVaeLoss, UniquenessLoss
from modules.quantize import Quantize, QuantizeForwardMode


torch.set_float32_matmul_precision("high")


# ============================================================
# RQ-VAE Output
# ============================================================

class RqVaeOutput(NamedTuple):
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    codebook_loss: Tensor
    commitment_loss: Tensor

    # --------------------------------------------------------
    # event-level C2 (event_ids를 준 경우에만 채워짐)
    #
    # event_codebook_loss   = mean_E ||sg(z(E)) - q2(E)||²
    # event_commitment_loss = mean_E ||z(E) - sg(q2(E))||²
    #
    # 이때 codebook_loss / commitment_loss(기사 단위)에는
    # Q2 항이 들어가지 않는다 (Q1 + Q3만).
    # --------------------------------------------------------
    event_codebook_loss: Optional[Tensor] = None
    event_commitment_loss: Optional[Tensor] = None
    event_unique_ids: Optional[Tensor] = None
    event_codes: Optional[Tensor] = None

    # 로그용 quantizer별 기사 평균 loss
    loss_components: Optional[dict] = None


# ============================================================
# RQ-VAE Loss Output
# ============================================================

class RqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    codebook_loss: Tensor
    commitment_loss: Tensor
    uniqueness_loss: Tensor
    rqvae_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor

    # event-level C2 학습에서 assertion / 로그에 사용
    sem_ids: Optional[Tensor] = None
    loss_components: Optional[dict] = None


# ============================================================
# RQ-VAE
# ============================================================

class RqVae(
    nn.Module,
    PyTorchModelHubMixin,
):

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        num_categories: int,
        c2_codebook_size: int = 256,
        c3_codebook_size: int = 256,
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.STE,
        lambda_rec: float = 1.0,
        lambda_cb: float = 1.0,
        lambda_com: float = 0.25,
        lambda_uniq: float = 0.1,
        uniqueness_margin: float = 0.5,
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.num_categories = num_categories
        self.c2_codebook_size = c2_codebook_size
        self.c3_codebook_size = c3_codebook_size

        # ----------------------------------------------------
        # Model config
        # ----------------------------------------------------

        self._config = {
            "input_dim": input_dim,
            "embed_dim": embed_dim,
            "hidden_dims": hidden_dims,
            "num_categories": num_categories,
            "c2_codebook_size": c2_codebook_size,
            "c3_codebook_size": c3_codebook_size,
            "codebook_normalize": codebook_normalize,
            "codebook_sim_vq": codebook_sim_vq,
            "codebook_mode": codebook_mode,
            "lambda_rec": lambda_rec,
            "lambda_cb": lambda_cb,
            "lambda_com": lambda_com,
            "lambda_uniq": lambda_uniq,
            "uniqueness_margin": uniqueness_margin,
        }

        # ----------------------------------------------------
        # Encoder
        #
        # x(a) ∈ R^input_dim
        #     ↓
        # h(a) ∈ R^embed_dim
        # ----------------------------------------------------

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=codebook_normalize,
        )

        # ----------------------------------------------------
        # Decoder
        #
        # q1 + q2 + q3
        #     ↓
        # x_hat ∈ R^input_dim
        # ----------------------------------------------------

        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims[-1::-1],
            out_dim=input_dim,
            normalize=False,
        )

        # ----------------------------------------------------
        # Q1
        #
        # C1 index:
        # category ID로 deterministic하게 지정
        #
        # Q1 vector:
        # trainable embedding
        # ----------------------------------------------------

        self.quantizer_1 = Quantize(
            embed_dim=embed_dim,
            n_embed=num_categories,
            codebook_normalize=codebook_normalize,
            sim_vq=codebook_sim_vq,
            forward_mode=codebook_mode,
        )

        # ----------------------------------------------------
        # Q2
        #
        # 학습:
        # r1 기준 quantization
        #
        # 최종 SID 생성:
        # event 단위로 미리 계산한 fixed_c2_ids를
        # 전달할 수도 있음
        # ----------------------------------------------------

        self.quantizer_2 = Quantize(
            embed_dim=embed_dim,
            n_embed=c2_codebook_size,
            codebook_normalize=codebook_normalize,
            sim_vq=codebook_sim_vq,
            forward_mode=codebook_mode,
        )

        # ----------------------------------------------------
        # Q3
        #
        # r2에 대해 article-level quantization
        # ----------------------------------------------------

        self.quantizer_3 = Quantize(
            embed_dim=embed_dim,
            n_embed=c3_codebook_size,
            codebook_normalize=codebook_normalize,
            sim_vq=codebook_sim_vq,
            forward_mode=codebook_mode,
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        self.reconstruction_loss_fn = ReconstructionLoss()

        self.uniqueness_loss_fn = UniquenessLoss(
            margin=uniqueness_margin,
        )

        self.loss_fn = RqVaeLoss(
            lambda_rec=lambda_rec,
            lambda_cb=lambda_cb,
            lambda_com=lambda_com,
            lambda_uniq=lambda_uniq,
        )


    # ========================================================
    # Properties
    # ========================================================

    @cached_property
    def config(self) -> dict:
        return self._config


    @property
    def device(self) -> torch.device:
        return next(
            self.encoder.parameters()
        ).device


    # ========================================================
    # Encoder / Decoder
    # ========================================================

    def encode(
        self,
        x: Tensor,
    ) -> Tensor:

        return self.encoder(x)


    def decode(
        self,
        x: Tensor,
    ) -> Tensor:

        return self.decoder(x)


    # ========================================================
    # Q2 Codebook Initialization
    # ========================================================

    @torch.no_grad()
    def set_c2_codebook(
        self,
        centroids: Tensor,
    ) -> None:

        expected_shape = (
            self.c2_codebook_size,
            self.embed_dim,
        )

        actual_shape = tuple(
            centroids.shape
        )

        if actual_shape != expected_shape:

            raise ValueError(
                "C2 centroid shape mismatch. "
                f"Expected {expected_shape}, "
                f"got {actual_shape}."
            )

        self.quantizer_2.set_codebook(
            centroids
        )


    # ========================================================
    # C2 Residual
    #
    # Event representation 계산 시 사용 가능
    #
    # x
    # ↓
    # Encoder
    # ↓
    # h
    # ↓
    # Q1[category]
    # ↓
    # r1 = h - q1
    #
    # Q2가 실제로 받는 입력은 r1이므로
    # event-level C2를 생성할 때도 r1을 평균낼 수 있음
    # ========================================================

    def get_c2_residual(
        self,
        x: Tensor,
        category_ids: Tensor,
        gumbel_t: float = 0.001,
    ) -> Tensor:

        x = x.to(
            device=self.device,
            dtype=next(
                self.encoder.parameters()
            ).dtype,
        )

        category_ids = category_ids.to(
            device=self.device,
            dtype=torch.long,
        )

        # Encoder
        h = self.encode(x)

        # Q1:
        # category ID가 C1 index를 결정
        q1_out = self.quantizer_1(
            x=h,
            temperature=gumbel_t,
            fixed_ids=category_ids,
        )

        q1 = q1_out.embeddings

        # Q2가 받는 residual
        r1 = h - q1

        return r1


    # ========================================================
    # Semantic ID Generation
    # ========================================================

    def get_semantic_ids(
        self,
        x: Tensor,
        category_ids: Tensor,
        gumbel_t: float = 0.001,
        fixed_c2_ids: Optional[Tensor] = None,
        event_ids: Optional[Tensor] = None,
        event_code_override: Optional[Tensor] = None,
    ) -> RqVaeOutput:

        # ----------------------------------------------------
        # Input device / dtype
        # ----------------------------------------------------

        x = x.to(
            device=self.device,
            dtype=next(
                self.encoder.parameters()
            ).dtype,
        )

        category_ids = category_ids.to(
            device=self.device,
            dtype=torch.long,
        )

        if fixed_c2_ids is not None:

            fixed_c2_ids = fixed_c2_ids.to(
                device=self.device,
                dtype=torch.long,
            )

            fixed_c2_ids = fixed_c2_ids.view(-1)

            if (
                fixed_c2_ids.shape[0]
                != x.shape[0]
            ):

                raise ValueError(
                    "fixed_c2_ids batch size "
                    "must match x. "
                    f"x batch={x.shape[0]}, "
                    f"fixed_c2_ids batch="
                    f"{fixed_c2_ids.shape[0]}."
                )

        if (
            event_ids is not None
            and fixed_c2_ids is not None
        ):
            raise ValueError(
                "event_ids and fixed_c2_ids "
                "cannot be used together."
            )

        # ----------------------------------------------------
        # Encoder
        #
        # x → h
        # ----------------------------------------------------

        h = self.encode(x)

        # ----------------------------------------------------
        # Q1
        #
        # c1 = category ID
        # q1 = Q1[c1]
        # ----------------------------------------------------

        q1_out = self.quantizer_1(
            x=h,
            temperature=gumbel_t,
            fixed_ids=category_ids,
        )

        q1 = q1_out.embeddings
        c1 = q1_out.ids

        # ----------------------------------------------------
        # Residual 1
        #
        # r1 = h - q1
        # ----------------------------------------------------

        r1 = h - q1

        # ----------------------------------------------------
        # Q2
        #
        # fixed_c2_ids == None
        # → 기존 방식
        # → r1에서 Q2 code 선택
        #
        # fixed_c2_ids != None
        # → event-level에서 미리 정해둔
        #   C2 index를 그대로 사용
        # ----------------------------------------------------

        event_codebook_loss = None
        event_commitment_loss = None
        event_unique_ids = None
        event_codes = None

        if event_ids is None:

            q2_out = self.quantizer_2(
                x=r1,
                temperature=gumbel_t,
                fixed_ids=fixed_c2_ids,
            )

            q2 = q2_out.embeddings
            c2 = q2_out.ids

            q2_codebook_loss = q2_out.codebook_loss
            q2_commitment_loss = q2_out.commitment_loss

        else:

            # ------------------------------------------------
            # event-level C2 (교수님 설계)
            #
            # c2 = EventCode[event(a)]
            # z(E) = mean{h(a) | a in E}
            # EventCode(E) = argmin_k ||z(E) - Q2[k]||
            # 같은 event의 기사는 같은 q2(E)를 공유
            # ------------------------------------------------

            (
                q2,
                c2,
                event_codebook_loss,
                event_commitment_loss,
                event_unique_ids,
                event_codes,
            ) = self._event_level_q2(
                h=h,
                r1=r1,
                event_ids=event_ids,
                event_code_override=event_code_override,
            )

            # Q2 loss는 event 단위로만 계산하므로
            # 기사 단위 합계에는 넣지 않는다.
            q2_codebook_loss = torch.zeros_like(
                q1_out.codebook_loss
            )
            q2_commitment_loss = torch.zeros_like(
                q1_out.commitment_loss
            )

        # ----------------------------------------------------
        # Residual 2
        #
        # r2 = r1 - q2
        #    = h - q1 - q2
        # ----------------------------------------------------

        r2 = r1 - q2

        # ----------------------------------------------------
        # Q3
        #
        # r2에서 article-level code 선택
        # ----------------------------------------------------

        q3_out = self.quantizer_3(
            x=r2,
            temperature=gumbel_t,
        )

        q3 = q3_out.embeddings
        c3 = q3_out.ids

        # ----------------------------------------------------
        # Quantization Loss
        # ----------------------------------------------------

        codebook_loss = (
            q1_out.codebook_loss
            + q2_codebook_loss
            + q3_out.codebook_loss
        )

        commitment_loss = (
            q1_out.commitment_loss
            + q2_commitment_loss
            + q3_out.commitment_loss
        )

        with torch.no_grad():
            loss_components = {
                "codebook_q1": q1_out.codebook_loss.mean(),
                "codebook_q2": (
                    event_codebook_loss
                    if event_codebook_loss is not None
                    else q2_codebook_loss.mean()
                ),
                "codebook_q3": q3_out.codebook_loss.mean(),
                "commitment_q1": q1_out.commitment_loss.mean(),
                "commitment_q2": (
                    event_commitment_loss
                    if event_commitment_loss is not None
                    else q2_commitment_loss.mean()
                ),
                "commitment_q3": q3_out.commitment_loss.mean(),
            }

        # ----------------------------------------------------
        # Quantized embeddings
        #
        # shape:
        # [batch, embed_dim, 3]
        # ----------------------------------------------------

        embeddings = torch.stack(
            [
                q1,
                q2,
                q3,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Residuals
        #
        # Q1 input = h
        # Q2 input = r1
        # Q3 input = r2
        # ----------------------------------------------------

        residuals = torch.stack(
            [
                h,
                r1,
                r2,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Semantic IDs
        #
        # (c1, c2, c3)
        # ----------------------------------------------------

        sem_ids = torch.stack(
            [
                c1,
                c2,
                c3,
            ],
            dim=-1,
        )

        return RqVaeOutput(
            embeddings=embeddings,
            residuals=residuals,
            sem_ids=sem_ids,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            event_codebook_loss=event_codebook_loss,
            event_commitment_loss=event_commitment_loss,
            event_unique_ids=event_unique_ids,
            event_codes=event_codes,
            loss_components=loss_components,
        )


    # ========================================================
    # Event-level Q2
    # ========================================================

    def _event_level_q2(
        self,
        h: Tensor,
        r1: Tensor,
        event_ids: Tensor,
        event_code_override: Optional[Tensor] = None,
    ):
        """
        batch 안의 event별로

            z(E) = mean{h(a) | a in E}            (현재 encoder output)
            EventCode(E) = argmin_k ||z(E) - Q2[k]||   (현재 Q2)
            q2(E) = Q2[EventCode(E)]

        를 계산하고 같은 event의 모든 기사에 q2(E)를 준다.
        batch에는 event의 모든 기사가 들어 있어야 한다
        (EventBatchSampler가 보장).

        event_code_override:
            기사별 code. -1이면 z(E) nearest로 정하고,
            0 이상이면 그 code를 그대로 쓴다
            (Validation에서 기존 Train event의 EventCode 상속).

        Gradient:
            학습 중 q2는 기존 Q2 STE와 같은 형태
                q2(a) = r1(a) + sg(q2(E) - r1(a))
            로 넘긴다. forward 값은 q2(E)이고, 기존 article-level STE와
            같은 gradient 경로를 유지한다 (Q3 commitment가 encoder /
            Q1 / Q2 codebook으로 새지 않음).
            Q2 codebook은 event_codebook_loss로만 학습된다.
        """

        quantizer = self.quantizer_2

        if (
            quantizer.forward_mode
            != QuantizeForwardMode.STE
        ):
            raise NotImplementedError(
                "event-level C2 currently supports "
                "QuantizeForwardMode.STE only."
            )

        event_ids = event_ids.to(
            device=h.device,
            dtype=torch.long,
        ).view(-1)

        if event_ids.shape[0] != h.shape[0]:
            raise ValueError(
                "event_ids batch size must match x. "
                f"x batch={h.shape[0]}, "
                f"event_ids batch={event_ids.shape[0]}."
            )

        (
            event_unique_ids,
            inverse,
        ) = torch.unique(
            event_ids,
            return_inverse=True,
        )

        num_events = event_unique_ids.shape[0]

        counts = torch.bincount(
            inverse,
            minlength=num_events,
        ).to(h.dtype).unsqueeze(1)

        # z(E) = mean h(a)  (gradient가 encoder로 흐름)
        z = torch.zeros(
            num_events,
            h.shape[1],
            device=h.device,
            dtype=h.dtype,
        ).index_add(
            0,
            inverse,
            h,
        ) / counts

        # EventCode(E) = current Q2 nearest
        with torch.no_grad():

            codebook = quantizer.out_proj(
                quantizer.embedding.weight
            )

            distances = quantizer._compute_distance(
                x=z.float(),
                codebook=codebook.float(),
            )

            event_codes = distances.argmin(dim=1)

            if event_code_override is not None:

                override = event_code_override.to(
                    device=h.device,
                    dtype=torch.long,
                ).view(-1)

                if override.shape[0] != h.shape[0]:
                    raise ValueError(
                        "event_code_override batch size "
                        "must match x."
                    )

                override_max = torch.full(
                    (num_events,),
                    -1,
                    device=h.device,
                    dtype=torch.long,
                ).scatter_reduce(
                    0,
                    inverse,
                    override,
                    reduce="amax",
                    include_self=True,
                )

                override_min = torch.full(
                    (num_events,),
                    quantizer.n_embed,
                    device=h.device,
                    dtype=torch.long,
                ).scatter_reduce(
                    0,
                    inverse,
                    override,
                    reduce="amin",
                    include_self=True,
                )

                if not torch.equal(
                    override_max,
                    override_min,
                ):
                    raise ValueError(
                        "event_code_override must be "
                        "identical within an event."
                    )

                event_codes = torch.where(
                    override_max >= 0,
                    override_max,
                    event_codes,
                )

        q2_event = quantizer.get_item_embeddings(
            event_codes
        )

        event_codebook_loss = (
            (z.detach() - q2_event)
            .pow(2)
            .sum(dim=-1)
            .mean()
        )

        event_commitment_loss = (
            (z - q2_event.detach())
            .pow(2)
            .sum(dim=-1)
            .mean()
        )

        q2_shared = q2_event[inverse]

        if quantizer.training:
            # 기존 Q2 STE와 같은 형태 (query = Q2 입력 r1)
            q2 = r1 + (q2_shared - r1).detach()
        else:
            q2 = q2_shared

        c2 = event_codes[inverse]

        return (
            q2,
            c2,
            event_codebook_loss,
            event_commitment_loss,
            event_unique_ids,
            event_codes,
        )


    # ========================================================
    # Training Forward
    # ========================================================

    def forward(
        self,
        x: Tensor,
        category_ids: Tensor,
        gumbel_t: float = 0.001,
        event_ids: Optional[Tensor] = None,
        event_code_override: Optional[Tensor] = None,
    ) -> RqVaeComputedLosses:

        # ----------------------------------------------------
        # 학습 단계에서는 fixed_c2_ids를 주지 않음
        #
        # 즉 Q2는 기존 학습 방식대로
        # r1에서 code를 선택
        # ----------------------------------------------------

        # event_ids를 주면 event-level C2로 학습한다
        # (c2 = EventCode[event(a)]).
        quantized = self.get_semantic_ids(
            x=x,
            category_ids=category_ids,
            gumbel_t=gumbel_t,
            fixed_c2_ids=None,
            event_ids=event_ids,
            event_code_override=event_code_override,
        )

        # ----------------------------------------------------
        # q1 + q2 + q3
        # ----------------------------------------------------

        quantized_embedding = (
            quantized
            .embeddings
            .sum(dim=-1)
        )

        # ----------------------------------------------------
        # Decoder
        #
        # q1 + q2 + q3
        # ↓
        # x_hat
        # ----------------------------------------------------

        x_hat = self.decode(
            quantized_embedding
        )

        # ----------------------------------------------------
        # Reconstruction Loss
        # ----------------------------------------------------

        reconstruction_loss = (
            self.reconstruction_loss_fn(
                x_hat=x_hat,
                x=x,
            )
        )

        # ----------------------------------------------------
        # Quantization Loss
        # ----------------------------------------------------

        codebook_loss = (
            quantized.codebook_loss
        )

        commitment_loss = (
            quantized.commitment_loss
        )

        # ----------------------------------------------------
        # Uniqueness Loss (HiD-VAE DUL)
        #
        # quantization 이전 latent h(a)는
        # residuals[..., 0]에 이미 저장되어 있음
        # (Q1 input = h)
        # ----------------------------------------------------

        h_pre_quantization = (
            quantized
            .residuals[..., 0]
        )

        uniqueness_loss = (
            self.uniqueness_loss_fn(
                z0=h_pre_quantization,
                sem_ids=quantized.sem_ids,
            )
        )

        # ----------------------------------------------------
        # Total Loss
        #
        # λ_rec * reconstruction
        # + λ_cb * codebook
        # + λ_com * commitment
        # + λ_uniq * uniqueness
        # ----------------------------------------------------

        total_loss_per_sample = (
            self.loss_fn(
                reconstruction_loss=(
                    reconstruction_loss
                ),
                codebook_loss=(
                    codebook_loss
                ),
                commitment_loss=(
                    commitment_loss
                ),
                uniqueness_loss=(
                    uniqueness_loss
                ),
            )
        )

        loss = (
            total_loss_per_sample
            .mean()
        )

        # ----------------------------------------------------
        # RQ-VAE Quantization Loss
        # ----------------------------------------------------

        rqvae_loss_per_sample = (
            self.loss_fn.lambda_cb
            * codebook_loss
            +
            self.loss_fn.lambda_com
            * commitment_loss
        )

        # ----------------------------------------------------
        # event-level Q2 loss
        #
        # + λ_cb  * mean_E ||sg(z(E)) - q2(E)||²
        # + λ_com * mean_E ||z(E) - sg(q2(E))||²
        # ----------------------------------------------------

        if quantized.event_codebook_loss is not None:

            event_rqvae_loss = (
                self.loss_fn.lambda_cb
                * quantized.event_codebook_loss
                + self.loss_fn.lambda_com
                * quantized.event_commitment_loss
            )

            loss = loss + event_rqvae_loss

            codebook_loss = (
                codebook_loss
                + quantized.event_codebook_loss
            )

            commitment_loss = (
                commitment_loss
                + quantized.event_commitment_loss
            )

            rqvae_loss_per_sample = (
                rqvae_loss_per_sample
                + event_rqvae_loss
            )

        # ----------------------------------------------------
        # Logging Metrics
        # ----------------------------------------------------

        with torch.no_grad():

            embs_norm = (
                quantized
                .embeddings
                .norm(
                    p=2,
                    dim=1,
                )
            )

            num_articles = (
                quantized
                .sem_ids
                .shape[0]
            )

            if num_articles == 0:

                p_unique_ids = torch.tensor(
                    0.0,
                    device=self.device,
                    dtype=torch.float32,
                )

            else:

                num_unique = (
                    torch.unique(
                        quantized.sem_ids,
                        dim=0,
                    )
                    .shape[0]
                )

                p_unique_ids = torch.tensor(
                    num_unique
                    / num_articles,
                    device=self.device,
                    dtype=torch.float32,
                )

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=(
                reconstruction_loss.mean()
            ),
            codebook_loss=(
                codebook_loss.mean()
            ),
            commitment_loss=(
                commitment_loss.mean()
            ),
            uniqueness_loss=(
                uniqueness_loss
            ),
            rqvae_loss=(
                rqvae_loss_per_sample.mean()
            ),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
            sem_ids=quantized.sem_ids,
            loss_components=quantized.loss_components,
        )


    # ========================================================
    # Load Pretrained
    # ========================================================

    def load_pretrained(
        self,
        path: str,
    ) -> None:

        state = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.load_state_dict(
            state["model"]
        )

        if "iter" in state:

            print(
                "Loaded RQ-VAE checkpoint "
                f"(iteration={state['iter']})"
            )

        else:

            print(
                "Loaded RQ-VAE checkpoint."
            )