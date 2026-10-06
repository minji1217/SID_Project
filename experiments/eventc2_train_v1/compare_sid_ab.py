"""같은 RQ-VAE checkpoint에서 만든 SID 결과들(A, B, B-r1 ...)을 첫 번째(A) 기준으로 비교한다.

입력 폴더는 generate_semantic_ids.py의 --output_dir이며 모두 읽기만 한다.

    python experiments/eventc2_train_v1/compare_sid_ab.py \
        --variant A=<A semantic_ids> \
        --variant B=<B semantic_ids> \
        --variant B-r1=<B-r1 semantic_ids> \
        --out-dir <exp>/reports/sid_ab \
        [--checkpoint <final checkpoint> --data-dir <RQ-VAE data_dir>]   # reconstruction 지표

(--a-dir / --b-dir는 --variant A= / --variant B=와 같다.)
--checkpoint/--data-dir를 주면 저장된 (c1,c2,c3)로
x_hat = decoder(Q1[c1] + Q2[c2] + Q3[c3])를 만들어 reconstruction을 따로 계산한다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


SID_FILES = {
    "train": "train_article_semantic_ids.parquet",
    "validation": "validation_article_semantic_ids.parquet",
    "all": "article_semantic_ids.parquet",
}


def load_sid_dir(sid_dir: Path) -> Dict[str, pd.DataFrame]:
    frames = {
        split: pd.read_parquet(sid_dir / filename)
        for split, filename in SID_FILES.items()
    }
    for frame in frames.values():
        frame["article_id"] = frame["article_id"].astype(str)

    frames["train_event_c2"] = pd.read_parquet(sid_dir / "train_event_c2_mapping.parquet")
    frames["validation_event_c2"] = pd.read_parquet(sid_dir / "validation_event_c2_mapping.parquet")
    return frames


def prefix_distribution(df: pd.DataFrame, columns: list[str]) -> Dict[str, Any]:
    sizes = df.groupby(columns).size().sort_values(ascending=False)
    probs = sizes / sizes.sum()
    return {
        "num_groups": int(len(sizes)),
        "entropy_bits": float(-(probs * np.log2(probs)).sum()),
        "max_group_size": int(sizes.iloc[0]),
        "median_group_size": float(sizes.median()),
        "top10_group_share": float(sizes.iloc[:10].sum() / sizes.sum()),
        "singleton_group_ratio": float((sizes == 1).mean()),
    }


def same_event_c2(df: pd.DataFrame) -> Dict[str, Any]:
    stats = df.groupby("event_id")["c2"].agg(
        size="size",
        nunique="nunique",
        dominant=lambda s: s.value_counts().iloc[0],
    )
    multi = stats[stats["size"] >= 2]

    if len(multi) == 0:
        return {"multi_article_events": 0}

    return {
        "multi_article_events": int(len(multi)),
        # 모든 기사가 같은 c2인 event 비율
        "all_same_c2_event_ratio": float((multi["nunique"] == 1).mean()),
        # eval_c2_event_consistency.py와 같은 정의 (event별 최빈 c2 비율의 평균)
        "dominant_c2_ratio_mean": float((multi["dominant"] / multi["size"]).mean()),
    }


def sid_metrics(frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    train = frames["train"]
    validation = frames["validation"]
    all_df = frames["all"]

    event_to_c2 = frames["train_event_c2"].set_index("event_id")["event_c2"]
    event_sizes = train.groupby("event_id").size()
    train_event_size = train["event_id"].map(event_sizes)
    train_match = train["event_id"].map(event_to_c2) == train["c2"]

    # Validation에서 Train event를 상속한 event:
    # 재사용 Train 기사 + validation-only 기사가 모두 같은 c2인지
    inherited_ids = set(
        frames["validation_event_c2"]
        .query("source == 'inherited_train_event'")["event_id"]
        .astype(int)
    )
    is_reused = validation["article_id"].isin(set(train["article_id"]))
    inherited_rows = validation[validation["event_id"].astype(int).isin(inherited_ids)]
    inherited_mixed = inherited_rows.assign(reused=is_reused[inherited_rows.index])
    mixed_events = inherited_mixed.groupby("event_id").filter(
        lambda g: g["reused"].any() and (~g["reused"]).any()
    )
    mixed_consistency = (
        float((mixed_events.groupby("event_id")["c2"].nunique() == 1).mean())
        if len(mixed_events) > 0
        else None
    )

    val_only = validation[~is_reused]
    val_only_event_c2 = (
        frames["validation_event_c2"].set_index("event_id")["event_c2"]
    )

    duplicated_c123 = all_df.duplicated(subset=["c1", "c2", "c3"], keep=False)

    return {
        "counts": {
            "train_articles": int(len(train)),
            "validation_articles": int(len(validation)),
            "validation_reused_train_articles": int(is_reused.sum()),
            "validation_only_articles": int((~is_reused).sum()),
            "unique_articles": int(len(all_df)),
            "train_events": int(train["event_id"].nunique()),
            "train_singleton_events": int((event_sizes == 1).sum()),
        },
        "unique_codes": {
            "c1": int(all_df["c1"].nunique()),
            "c2": int(all_df["c2"].nunique()),
            "c3": int(all_df["c3"].nunique()),
            "c12": int(all_df[["c1", "c2"]].drop_duplicates().shape[0]),
            "c123": int(all_df[["c1", "c2", "c3"]].drop_duplicates().shape[0]),
            "c1234": int(all_df[["c1", "c2", "c3", "c4"]].drop_duplicates().shape[0]),
        },
        "collision": {
            # 다른 기사와 (c1,c2,c3)를 공유하는 기사 비율
            "c123_collision_article_ratio": float(duplicated_c123.mean()),
            # 1 - unique c123 / articles
            "c123_collision_rate": float(
                1 - all_df[["c1", "c2", "c3"]].drop_duplicates().shape[0] / len(all_df)
            ),
            "max_c4": int(all_df["c4"].max()),
        },
        "same_event_c2": {
            "train": same_event_c2(train),
            "validation": same_event_c2(validation),
            "all_unique_articles": same_event_c2(all_df),
        },
        "train_c2_equals_event_c2": {
            "all": float(train_match.mean()),
            "singleton_events": float(train_match[train_event_size == 1].mean()),
            "multi_article_events": float(train_match[train_event_size >= 2].mean()),
        },
        "validation_inheritance": {
            "inherited_train_events": int(len(inherited_ids)),
            "events_with_reused_and_new_articles": int(mixed_events["event_id"].nunique()),
            "reused_and_new_same_c2_event_ratio": mixed_consistency,
            "validation_only_c2_equals_event_c2": float(
                (val_only["event_id"].map(val_only_event_c2) == val_only["c2"]).mean()
            ) if len(val_only) > 0 else None,
        },
        "prefix2_c1c2": prefix_distribution(all_df, ["c1", "c2"]),
        "prefix3_c1c2c3": prefix_distribution(all_df, ["c1", "c2", "c3"]),
        "train_event_size": {
            "max": int(event_sizes.max()),
            "p99": float(event_sizes.quantile(0.99)),
            "mean": float(event_sizes.mean()),
            "top5": [int(v) for v in event_sizes.sort_values(ascending=False).iloc[:5]],
        },
    }


def change_metrics(base: Dict[str, pd.DataFrame], other: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """baseline(A) 대비 기사별 code 변경. Train은 singleton / multi-article event로 나눠 센다."""
    result: Dict[str, Any] = {}
    for split in ("train", "validation", "all"):
        merged = base[split].merge(
            other[split], on="article_id", suffixes=("_a", "_b"), how="inner", validate="one_to_one"
        )
        if len(merged) != len(base[split]) or len(merged) != len(other[split]):
            raise RuntimeError(f"{split}: article set이 다릅니다.")

        entry: Dict[str, Any] = {"articles": int(len(merged))}
        for col in ("c1", "c2", "c3", "c4"):
            changed = merged[f"{col}_a"] != merged[f"{col}_b"]
            entry[f"{col}_changed_count"] = int(changed.sum())
            entry[f"{col}_changed_ratio"] = float(changed.mean())

        if split == "train":
            event_size = merged.groupby("event_id_a")["article_id"].transform("size")
            c2_changed = merged["c2_a"] != merged["c2_b"]
            for name, mask in (("singleton", event_size == 1), ("multi_article", event_size >= 2)):
                entry[f"{name}_event_articles"] = int(mask.sum())
                entry[f"{name}_event_c2_changed_count"] = int((c2_changed & mask).sum())
                entry[f"{name}_event_c2_changed_ratio"] = (
                    float((c2_changed & mask).sum() / mask.sum()) if mask.sum() > 0 else None
                )

        result[split] = entry
    return result


def reconstruction_metrics(
    frames: Dict[str, pd.DataFrame],
    checkpoint: Path,
    data_dir: Path,
) -> Dict[str, Any]:
    import torch

    rqvae_dir = Path(__file__).resolve().parents[2] / "RQVAE"
    sys.path.insert(0, str(rqvae_dir))
    from generate_semantic_ids import load_rqvae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_rqvae(checkpoint_path=str(checkpoint), device=device)
    embeddings = np.load(data_dir / "article_embeddings.npy", mmap_mode="r")

    result = {}
    for split in ("train", "validation"):
        df = frames[split]
        rec_sum = 0.0
        r2_sum = 0.0
        r3_sum = 0.0
        for start in range(0, len(df), 4096):
            chunk = df.iloc[start:start + 4096]
            x = torch.from_numpy(
                np.asarray(embeddings[chunk["embedding_row"].to_numpy()], dtype=np.float32)
            ).to(device)
            ids = {
                col: torch.tensor(chunk[col].to_numpy(), dtype=torch.long, device=device)
                for col in ("c1", "c2", "c3")
            }
            with torch.no_grad():
                h = model.encode(x)
                q1 = model.quantizer_1.get_item_embeddings(ids["c1"])
                q2 = model.quantizer_2.get_item_embeddings(ids["c2"])
                q3 = model.quantizer_3.get_item_embeddings(ids["c3"])
                x_hat = model.decode(q1 + q2 + q3)
                rec_sum += float(((x_hat - x) ** 2).sum(dim=-1).sum())
                r2_sum += float((h - q1 - q2).norm(dim=-1).sum())
                r3_sum += float((h - q1 - q2 - q3).norm(dim=-1).sum())
        result[split] = {
            "reconstruction_loss": rec_sum / len(df),
            "r2_norm_mean": r2_sum / len(df),
            "final_residual_norm_mean": r3_sum / len(df),
        }
    return result


def flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, inner, out)
    else:
        out[prefix] = value


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


VARIANT_ROLES = {
    "A": "기존: article-level Q2 nearest (r1 = h - q1)",
    "B": "교수님 설계: event-level Q2(mean(h))",
    "B-r1": "진단용: event-level Q2(mean(h - q1))",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="SID 비교 (A 대비 B, B-r1 등)")
    parser.add_argument(
        "--variant", action="append", default=[], metavar="NAME=DIR",
        help="비교할 SID 폴더. 첫 번째가 기준(A). 예: --variant A=dir --variant B=dir",
    )
    parser.add_argument("--a-dir", type=Path, default=None, help="--variant A=DIR와 같음")
    parser.add_argument("--b-dir", type=Path, default=None, help="--variant B=DIR와 같음")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    variants: list[tuple[str, Path]] = []
    if args.a_dir is not None:
        variants.append(("A", args.a_dir))
    if args.b_dir is not None:
        variants.append(("B", args.b_dir))
    for item in args.variant:
        name, _, path = item.partition("=")
        variants.append((name, Path(path)))

    if len(variants) < 2:
        parser.error("비교할 SID 폴더가 두 개 이상 필요합니다.")

    names = [name for name, _ in variants]
    frames = {name: load_sid_dir(path) for name, path in variants}
    base_name = names[0]

    report: Dict[str, Any] = {
        "variants": {name: {"dir": str(path), "role": VARIANT_ROLES.get(name)} for name, path in variants},
        "baseline": base_name,
        "sid_metrics": {name: sid_metrics(frames[name]) for name in names},
        "change_vs_baseline": {
            name: change_metrics(frames[base_name], frames[name]) for name in names[1:]
        },
    }

    if args.checkpoint is not None and args.data_dir is not None:
        report["reconstruction"] = {
            name: reconstruction_metrics(frames[name], args.checkpoint, args.data_dir) for name in names
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sid_ab_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def section(title: str, columns: list[str], per_column: Dict[str, Any]) -> list[str]:
        flat = {}
        for column in columns:
            flat[column] = {}
            flatten("", per_column[column], flat[column])
        keys = list(flat[columns[0]])
        lines = [f"### {title}", "", "| metric | " + " | ".join(columns) + " |",
                 "|---|" + "---|" * len(columns)]
        lines += [
            f"| {key} | " + " | ".join(fmt(flat[column].get(key)) for column in columns) + " |"
            for key in keys
        ]
        return lines + [""]

    lines = ["| variant | role |", "|---|---|"]
    lines += [f"| {name} | {VARIANT_ROLES.get(name, '-')} |" for name in names]
    lines.append("")
    lines += section("SID 지표", names, report["sid_metrics"])
    lines += section(f"{base_name} 대비 변경", names[1:], report["change_vs_baseline"])
    if "reconstruction" in report:
        lines += section("reconstruction (별도 지표, 저장된 c1,c2,c3로 decode)", names, report["reconstruction"])

    table = "\n".join(lines)
    (args.out_dir / "sid_ab_report.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nsaved: {args.out_dir}")


if __name__ == "__main__":
    main()
