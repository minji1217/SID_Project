"""event-level C2 구현 검증 (합성 데이터, CPU)

    cd RQVAE
    python tests/check_event_c2.py

확인 항목
  1. EventBatchSampler: event 분할 없음, 모든 기사 1회, seed 재현성, epoch별 순서 변화
  2. event mode forward
     - 같은 event의 c2 동일, EventCode = argmin ||mean h - Q2||
     - L_cb2 / L_com2 = 손 계산값
     - event_code_override (Validation 상속) 적용과 event 내부 불일치 거부
     - eval 모드 q2 = Q2[EventCode], train 모드 forward 값도 Q2[EventCode]
  3. gradient 경로
     - reconstruction -> codebook(Q1/Q2/Q3)로 0
     - Q3 commitment -> encoder / Q1 / Q2로 0 (기존 article-level STE와 동일)
     - L_cb2 -> Q2 codebook만, L_com2 -> encoder만
"""

import sys

from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.event_batch_sampler import EventBatchSampler  # noqa: E402
from event_c2_utils import assert_one_c2_per_event  # noqa: E402
from modules.rqvae import RqVae  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


def make_events(rng, num_events=300, max_size=12):
    sizes = np.minimum(rng.geometric(0.5, num_events), max_size)
    event_ids = np.repeat(np.arange(num_events) * 7 + 3, sizes)
    return event_ids[rng.permutation(len(event_ids))]


def check_sampler():
    print("[1] EventBatchSampler")
    rng = np.random.default_rng(0)
    event_ids = make_events(rng)

    sampler = EventBatchSampler(event_ids, batch_size=64, shuffle=True, seed=42)
    orders = []

    for epoch in range(3):
        sampler.set_epoch(epoch)
        batches = list(sampler)
        flat = np.concatenate(batches)

        check(
            sorted(flat.tolist()) == list(range(len(event_ids))),
            f"epoch {epoch}: every article exactly once",
        )

        batch_of = {}
        for b, batch in enumerate(batches):
            for index in batch:
                batch_of.setdefault(event_ids[index], set()).add(b)

        check(
            all(len(v) == 1 for v in batch_of.values()),
            f"epoch {epoch}: no event split across batches",
        )
        check(
            all(len(batch) <= 64 for batch in batches),
            f"epoch {epoch}: batch size <= 64",
        )
        orders.append(flat.tolist())

    check(orders[0] != orders[1], "event order changes across epochs")

    again = EventBatchSampler(event_ids, batch_size=64, shuffle=True, seed=42)
    again.set_epoch(1)
    check(np.concatenate(list(again)).tolist() == orders[1], "same seed/epoch -> same batches")


def make_model():
    torch.manual_seed(0)
    model = RqVae(
        input_dim=32,
        embed_dim=16,
        hidden_dims=[24],
        num_categories=5,
        c2_codebook_size=12,
        c3_codebook_size=20,
        lambda_cb=0.25,
        lambda_com=0.1,
        lambda_uniq=0.05,
        uniqueness_margin=0.5,
    )
    with torch.no_grad():
        model.quantizer_1.embedding.weight.normal_(0, 0.5)
        model.quantizer_2.embedding.weight.normal_(0, 0.5)
        model.quantizer_3.embedding.weight.normal_(0, 0.3)
    return model


def make_batch(n=60):
    rng = np.random.default_rng(1)
    event_ids = torch.tensor(
        np.repeat(np.arange(20) + 100, rng.integers(1, 6, 20))[:n]
    )
    x = torch.randn(len(event_ids), 32)
    category_ids = torch.randint(0, 5, (len(event_ids),))
    return x, category_ids, event_ids


def check_forward():
    print("[2] event mode forward")
    model = make_model()
    x, category_ids, event_ids = make_batch()

    for training in (False, True):
        model.train(training)
        out = model.get_semantic_ids(x, category_ids, event_ids=event_ids)
        c2 = out.sem_ids[:, 1]
        assert_one_c2_per_event(event_ids, c2)

        with torch.no_grad():
            h = model.encode(x)
            unique_ids, inverse = torch.unique(event_ids, return_inverse=True)
            z = torch.stack([h[inverse == i].mean(0) for i in range(len(unique_ids))])
            codes = torch.cdist(z, model.quantizer_2.embedding.weight).argmin(1)
            q2e = model.quantizer_2.embedding.weight[codes]

        mode = "train" if training else "eval"
        check(torch.equal(c2, codes[inverse]), f"{mode}: c2 = argmin ||mean h - Q2|| shared in event")
        check(
            torch.allclose(out.event_codebook_loss, ((z - q2e) ** 2).sum(-1).mean(), atol=1e-6),
            f"{mode}: L_cb2 = mean_E ||z(E) - q2(E)||^2",
        )
        check(
            torch.allclose(out.event_commitment_loss, out.event_codebook_loss),
            f"{mode}: L_com2 value = L_cb2 value",
        )
        check(
            torch.allclose(out.embeddings[..., 1], q2e[inverse], atol=1e-6),
            f"{mode}: forward q2 = Q2[EventCode(E)]",
        )

    model.eval()
    override = torch.full_like(event_ids, -1)
    target_event = event_ids[0]
    override[event_ids == target_event] = 7
    out = model.get_semantic_ids(x, category_ids, event_ids=event_ids, event_code_override=override)
    check(
        bool((out.sem_ids[event_ids == target_event, 1] == 7).all()),
        "override: inherited event uses given code",
    )
    base = model.get_semantic_ids(x, category_ids, event_ids=event_ids)
    others = event_ids != target_event
    check(
        torch.equal(out.sem_ids[others, 1], base.sem_ids[others, 1]),
        "override: other events unchanged",
    )

    bad = override.clone()
    bad[(event_ids == target_event).nonzero()[0]] = 3
    try:
        model.get_semantic_ids(x, category_ids, event_ids=event_ids, event_code_override=bad)
    except ValueError:
        check(True, "override: inconsistent code within event is rejected")
    else:
        raise AssertionError("inconsistent override was accepted")


def grads(model, loss):
    groups = {
        "encoder": model.encoder,
        "Q1": model.quantizer_1,
        "Q2": model.quantizer_2,
        "Q3": model.quantizer_3,
        "decoder": model.decoder,
    }
    params = [p for g in groups.values() for p in g.parameters()]
    values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    result = {}
    i = 0
    for name, group in groups.items():
        n = len(list(group.parameters()))
        result[name] = sum(float(v.abs().sum()) for v in values[i:i + n] if v is not None)
        i += n
    return result


def check_gradients():
    print("[3] gradient paths (train mode)")
    model = make_model()
    model.train()
    x, category_ids, event_ids = make_batch()

    out = model.get_semantic_ids(x, category_ids, event_ids=event_ids)
    x_hat = model.decode(out.embeddings.sum(-1))
    reconstruction = ((x_hat - x) ** 2).sum(-1).mean()

    g = grads(model, reconstruction)
    check(g["encoder"] > 0 and g["Q1"] == g["Q2"] == g["Q3"] == 0, "reconstruction -> encoder only (no codebook)")

    g = grads(model, out.loss_components["commitment_q3"] * 0 + (
        (out.residuals[..., 2] - model.quantizer_3.embedding.weight[out.sem_ids[:, 2]].detach()) ** 2
    ).sum(-1).mean())
    check(g["encoder"] == g["Q1"] == g["Q2"] == 0, "Q3 commitment -> no encoder/Q1/Q2 (same as article STE)")

    g = grads(model, out.event_codebook_loss)
    check(g["Q2"] > 0 and g["encoder"] == g["Q1"] == g["Q3"] == 0, "L_cb2 -> Q2 codebook only")

    g = grads(model, out.event_commitment_loss)
    check(g["encoder"] > 0 and g["Q1"] == g["Q2"] == g["Q3"] == 0, "L_com2 -> encoder only")

    full = model(x, category_ids, 0.5, event_ids=event_ids)
    expected = full.loss  # recompute pieces to confirm composition
    per_article = (
        full.reconstruction_loss
        + 0.25 * (out.loss_components["codebook_q1"] + out.loss_components["codebook_q3"])
        + 0.1 * (out.loss_components["commitment_q1"] + out.loss_components["commitment_q3"])
        + 0.05 * full.uniqueness_loss
    )
    event_terms = 0.25 * out.event_codebook_loss + 0.1 * out.event_commitment_loss
    check(
        torch.allclose(expected, per_article + event_terms, atol=1e-5),
        "total = mean_a[rec + cb(q1,q3) + com(q1,q3)] + uniq + λcb·L_cb2 + λcom·L_com2",
    )


if __name__ == "__main__":
    check_sampler()
    check_forward()
    check_gradients()
    print("\nall checks passed")
