"""Step 2 of the input-check study: compare candidate gates on the same images and write the results.

Reads the images the model sees from .work/gate/<set>.npz (scripts/gate_extract.py) and compares

* zero-shot image-text gates from the CLIP family (several candidate models). The score of a photo is the
  softmax mass on the in-domain prompts, softmax over all prompts in
  src/poultry_disease/data/input_check_prompts.json, at the model's own logit scale;
* the same models with image prototypes (the mean embedding of each class in a seeded training sample),
  scored as the highest cosine similarity to a class prototype; with k-nearest-neighbour similarity (k = 10)
  to the same training sample; and a fused score, the mean of the z-scored zero-shot log-odds and kNN
  similarity (means and standard deviations from the validation images);
* the feature-distance baseline on the published ResNet50: Mahalanobis distance (class means, shared
  Ledoit-Wolf covariance) and k-nearest-neighbour distance (k = 10, cosine) to the same training sample.

Threshold rule, fixed before any test image was scored: the highest threshold that keeps the false-rejection
rate on the validation images (tuning set) at or below 1%. The out-of-distribution (OOD) photos are split
once into ood-tune and ood-test (docs/ood/split.json). Decisions use the tuning sets; test numbers are
reported, not used to choose.

For each zero-shot candidate the image encoder is also exported to ONNX (fp32, fp16, int8 weights) to
measure download size, CPU latency (ONNX Runtime) and agreement with the PyTorch embeddings.

Output: results/input_check/candidate_study.json and .md. Run with the gate venv:

    .work/venv-gate/Scripts/python scripts/gate_study.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".work" / "gate"
OUT = ROOT / "results" / "input_check"
PROMPTS = ROOT / "src" / "poultry_disease" / "data" / "input_check_prompts.json"
SPLIT = ROOT / "docs" / "ood" / "split.json"
CREDITS = ROOT / "docs" / "ood" / "credits.json"
MAX_FRR = 0.01
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

#: name -> how to load it, and the licence as read from the model card and licence file on 2026-11-12.
CANDIDATES: dict[str, dict] = {
    "clip-vit-b32-openai": {
        "kind": "hf-clip",
        "repo": "openai/clip-vit-base-patch32",
        "license": "MIT (github.com/openai/CLIP LICENSE); model card: any deployed use is out of scope",
    },
    "openclip-vit-b32-laion2b": {
        "kind": "open_clip",
        "arch": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "repo": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "license": "MIT (model card); model card: any deployed use is out of scope",
    },
    "siglip-base-p16-224": {
        "kind": "hf-siglip",
        "repo": "google/siglip-base-patch16-224",
        "license": "Apache-2.0 (model card)",
    },
    "tinyclip-vit-40m-32-laion400m": {
        "kind": "hf-clip",
        "repo": "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
        "license": "MIT (model card; microsoft/Cream TinyCLIP/LICENSE)",
    },
    "tinyclip-vit-39m-16-yfcc15m": {
        "kind": "hf-clip",
        "repo": "wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M",
        "license": "MIT (model card; microsoft/Cream TinyCLIP/LICENSE)",
    },
    "tinyclip-vit-8m-16-yfcc15m": {
        "kind": "hf-clip",
        "repo": "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        "license": "MIT (model card; microsoft/Cream TinyCLIP/LICENSE)",
    },
}

# ----------------------------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": k / n if n else float("nan"), "ci95": [lo, hi]}


def threshold_for(scores: np.ndarray, max_frr: float = MAX_FRR) -> float:
    """Highest t such that the share of in-domain tuning scores below t is at most max_frr (pass: s >= t)."""
    s = np.sort(scores)
    return float(s[int(math.floor(max_frr * len(s)))])


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of an in-domain image > score of an OOD photo), ties count half."""
    allv = np.concatenate([pos, neg])
    ranks = np.empty(len(allv))
    order = np.argsort(allv, kind="mergesort")
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# --------------------------------------------------------------------------------------------------- data


def load_set(name: str) -> tuple[dict, list[dict]]:
    data = dict(np.load(WORK / f"{name}.npz"))
    meta = json.loads((WORK / f"{name}.json").read_text("utf-8"))
    return data, meta


def ood_groups() -> tuple[dict[str, str], dict[str, str], set[str], set[str]]:
    credits = json.loads(CREDITS.read_text("utf-8"))["photos"]
    split = json.loads(SPLIT.read_text("utf-8"))
    category = {p["file"]: p["category"] for p in credits}
    proximity = {p["file"]: p["proximity"] for p in credits}
    return category, proximity, set(split["tune"]), set(split["test"])


# ------------------------------------------------------------------------------------------------- models


def load_candidate(name: str):
    """-> (torch image encoder taking float NHWC 0..255 and returning unit vectors, text fn, logit scale)."""
    import torch

    c = CANDIDATES[name]
    if c["kind"] == "hf-clip":
        from transformers import CLIPModel, CLIPTokenizerFast

        model = CLIPModel.from_pretrained(c["repo"]).eval()
        tok = CLIPTokenizerFast.from_pretrained(c["repo"])
        mean, std = CLIP_MEAN, CLIP_STD
        encode_image = model.get_image_features

        def encode_text(prompts):
            return model.get_text_features(**tok(prompts, padding=True, return_tensors="pt"))

        scale = float(model.logit_scale.exp())
        size = model.config.vision_config.image_size
    elif c["kind"] == "hf-siglip":
        from transformers import AutoTokenizer, SiglipModel

        model = SiglipModel.from_pretrained(c["repo"]).eval()
        tok = AutoTokenizer.from_pretrained(c["repo"])
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        encode_image = model.get_image_features

        def encode_text(prompts):
            return model.get_text_features(**tok(prompts, padding="max_length", return_tensors="pt"))

        scale = float(model.logit_scale.exp())
        size = model.config.vision_config.image_size
    elif c["kind"] == "open_clip":
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(c["arch"], pretrained=c["pretrained"])
        model.eval()
        tok = open_clip.get_tokenizer(c["arch"])
        mean, std = CLIP_MEAN, CLIP_STD
        encode_image = model.encode_image

        def encode_text(prompts):
            return model.encode_text(tok(prompts))

        scale = float(model.logit_scale.exp())
        size = 224
    else:
        raise ValueError(c["kind"])
    if size != 224:
        raise ValueError(f"{name}: expects {size} px input, the gate reads the 224 px image the model sees")

    class ImageEncoder(torch.nn.Module):
        """float32 [N, 224, 224, 3] in 0..255 (the image the classifier sees) -> unit-length embedding."""

        def __init__(self):
            super().__init__()
            self.inner = model
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255.0)
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255.0)

        def forward(self, x):
            x = (x.permute(0, 3, 1, 2) - self.mean) / self.std
            f = encode_image(x)
            return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def text(prompts):
        f = encode_text(prompts)
        return (f / f.norm(dim=-1, keepdim=True)).numpy().astype(np.float32)

    return ImageEncoder().eval(), text, scale


def embed(encoder, seen: np.ndarray, batch: int = 64) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(seen), batch):
            out.append(encoder(torch.from_numpy(seen[i : i + batch].astype(np.float32))).numpy())
    return np.concatenate(out).astype(np.float32)


def zero_shot_score(img: np.ndarray, text_id: np.ndarray, text_rej: np.ndarray, scale: float) -> np.ndarray:
    logits = scale * img @ np.concatenate([text_id, text_rej]).T
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p[:, : len(text_id)].sum(axis=1)


def prototype_score(img: np.ndarray, protos: np.ndarray) -> np.ndarray:
    return (img @ protos.T).max(axis=1)


# ---------------------------------------------------------------------------------------- ONNX measurements


def onnx_measurements(name: str, encoder, ref_seen: np.ndarray, ref_emb: np.ndarray) -> dict:
    import onnx
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.transformers.float16 import convert_float_to_float16

    d = ROOT / "build" / "gate" / name / "onnx"
    d.mkdir(parents=True, exist_ok=True)
    fp32 = d / "image_encoder.fp32.onnx"
    if not fp32.exists():
        torch.onnx.export(
            encoder,
            torch.zeros(1, 224, 224, 3),
            str(fp32),
            input_names=["image"],
            output_names=["embedding"],
            dynamic_axes={"image": {0: "n"}, "embedding": {0: "n"}},
            opset_version=17,
        )
    fp16 = d / "image_encoder.fp16.onnx"
    if not fp16.exists():
        m = convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True)
        onnx.save(m, str(fp16))
    int8 = d / "image_encoder.int8.onnx"
    if not int8.exists():
        # MatMul weights only: ONNX Runtime has no CPU kernel for ConvInteger (the quantised patch embedding)
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])

    out = {}
    x1 = ref_seen[:1].astype(np.float32)
    for label, path in (("fp32", fp32), ("fp16", fp16), ("int8", int8)):
        row = {"bytes": path.stat().st_size}
        for threads in (1, 4):
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            t0 = time.perf_counter()
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            load = time.perf_counter() - t0
            sess.run(None, {"image": x1})
            times = []
            for _ in range(15):
                t0 = time.perf_counter()
                sess.run(None, {"image": x1})
                times.append(time.perf_counter() - t0)
            row[f"cpu_{threads}t_load_s"] = load
            row[f"cpu_{threads}t_median_ms"] = 1000 * float(np.median(times))
        emb = np.concatenate(
            [sess.run(None, {"image": ref_seen[i : i + 16].astype(np.float32)})[0] for i in range(0, len(ref_seen), 16)]
        )
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        cos = (emb * ref_emb).sum(axis=1)
        row["cosine_vs_pytorch_min"] = float(cos.min())
        row["cosine_vs_pytorch_mean"] = float(cos.mean())
        out[label] = row
    return out


# -------------------------------------------------------------------------------------------------- report


def evaluate(scores: dict[str, np.ndarray], meta: dict[str, list[dict]], groups) -> dict:
    """Threshold from val, then every rate on the tuning and test sets."""
    category, proximity, tune, test = groups
    t = threshold_for(scores["val"])
    res: dict = {"threshold": t}

    def frr(name):
        s, labels = scores[name], [m["label"] for m in meta[name]]
        by = {
            c: rate(int(sum(1 for x, lab in zip(s, labels, strict=True) if lab == c and x < t)), labels.count(c))
            for c in sorted(set(labels))
        }
        return {"overall": rate(int((s < t).sum()), len(s)), "by_class": by}

    res["frr_val"] = frr("val")
    res["frr_test_internal"] = frr("test_internal")
    res["frr_test_external"] = frr("test_external")
    both = np.concatenate([scores["test_internal"], scores["test_external"]])
    res["frr_test_all"] = rate(int((both < t).sum()), len(both))
    res["samples_failing"] = [m["id"] for m, x in zip(meta["samples"], scores["samples"], strict=True) if x < t]

    ids = [m["id"] for m in meta["ood"]]
    s_ood = scores["ood"]
    for part, members in (("tune", tune), ("test", test)):
        idx = [i for i, f in enumerate(ids) if f in members]
        acc = {
            "overall": rate(sum(s_ood[i] >= t for i in idx), len(idx)),
            "near": rate(
                sum(s_ood[i] >= t for i in idx if proximity[ids[i]] == "near"),
                sum(proximity[ids[i]] == "near" for i in idx),
            ),
            "far": rate(
                sum(s_ood[i] >= t for i in idx if proximity[ids[i]] == "far"),
                sum(proximity[ids[i]] == "far" for i in idx),
            ),
            "by_category": {
                c: rate(
                    sum(s_ood[i] >= t for i in idx if category[ids[i]] == c), sum(category[ids[i]] == c for i in idx)
                )
                for c in sorted({category[ids[i]] for i in idx})
            },
            "accepted": sorted(ids[i] for i in idx if s_ood[i] >= t),
        }
        res[f"far_ood_{part}"] = acc
        res[f"auroc_{'val' if part == 'tune' else 'test'}_vs_ood_{part}"] = auroc(
            scores["val" if part == "tune" else "test_all"], s_ood[idx]
        )
        near_idx = [i for i in idx if proximity[ids[i]] == "near"]
        pos = scores["val" if part == "tune" else "test_all"]
        res[f"auroc_near_{part}"] = auroc(pos, s_ood[near_idx])
        # ROC curve at 201 thresholds (score quantiles): false-rejection vs false-acceptance rates
        grid = np.unique(np.quantile(np.concatenate([pos, s_ood[idx]]), np.linspace(0, 1, 201)))
        res[f"roc_{part}"] = {
            "threshold": grid.tolist(),
            "frr": [float((pos < g).mean()) for g in grid],
            "far": [float((s_ood[idx] >= g).mean()) for g in grid],
            "far_near": [float((s_ood[near_idx] >= g).mean()) for g in grid],
        }
    return res


def make_scorers(emb: dict, t_id: np.ndarray, t_rej: np.ndarray, scale: float, protos: np.ndarray):
    """The four scores of a zero-shot candidate, plus the validation statistics of the fused score."""
    bank_e = emb["train_sample"]

    def knn_e(e, k=10):
        return np.sort(e @ bank_e.T, axis=1)[:, -k]

    def logodds(e):
        z = np.clip(zero_shot_score(e, t_id, t_rej, scale), 1e-7, 1 - 1e-7)
        return np.log(z / (1 - z))

    # fused: mean of the z-scored zero-shot log-odds and kNN similarity (z-scores from validation statistics)
    lv, kv = logodds(emb["val"]), knn_e(emb["val"])
    stats = {
        "logodds_mean": float(lv.mean()),
        "logodds_std": float(lv.std()),
        "knn_mean": float(kv.mean()),
        "knn_std": float(kv.std()),
    }

    def fused(e):
        return 0.5 * (
            (logodds(e) - stats["logodds_mean"]) / stats["logodds_std"]
            + (knn_e(e) - stats["knn_mean"]) / stats["knn_std"]
        )

    return {
        "zero_shot": lambda e: zero_shot_score(e, t_id, t_rej, scale),
        "prototype": lambda e: prototype_score(e, protos),
        "knn10": knn_e,
        "fused": fused,
    }, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", action="append", choices=sorted(CANDIDATES), help="default: all")
    ap.add_argument("--no-onnx", action="store_true", help="skip the ONNX size and latency measurements")
    args = ap.parse_args()

    sets = ["val", "train_sample", "test_internal", "test_external", "samples", "ood"]
    data, meta = {}, {}
    for s in sets:
        data[s], meta[s] = load_set(s)
    groups = ood_groups()
    prompts = json.loads(PROMPTS.read_text("utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "candidate_study.json"
    doc = json.loads(out_json.read_text("utf-8")) if out_json.exists() else {}
    doc["description"] = (
        "Input-check candidate study on poultry droppings. Threshold: highest score with at most 1% false "
        "rejections on the 973 validation images. ood-tune and ood-test from docs/ood/split.json. Test sets were "
        "not used to choose the threshold. Written by scripts/gate_study.py."
    )
    doc["prompts"] = prompts
    doc.setdefault("candidates", {})

    # feature-distance baseline on the published ResNet50
    from sklearn.covariance import LedoitWolf

    bank, labels = data["train_sample"]["feat"], [m["label"] for m in meta["train_sample"]]
    classes = sorted(set(labels))
    means = np.stack([bank[[lab == c for lab in labels]].mean(axis=0) for c in classes])
    cov = LedoitWolf().fit(bank - means[[classes.index(lab) for lab in labels]])
    prec = cov.precision_
    bank_n = bank / np.linalg.norm(bank, axis=1, keepdims=True)

    def maha(f):
        d = np.stack([np.einsum("ij,jk,ik->i", f - m, prec, f - m) for m in means], axis=1)
        return -d.min(axis=1)

    def knn(f, k=10):
        fn = f / np.linalg.norm(f, axis=1, keepdims=True)
        return -np.sort(1 - fn @ bank_n.T, axis=1)[:, k - 1]

    for label, fn in (("resnet50-mahalanobis", maha), ("resnet50-knn10", knn)):
        scores = {s: fn(data[s]["feat"]) for s in sets if s != "train_sample"}
        scores["test_all"] = np.concatenate([scores["test_internal"], scores["test_external"]])
        doc["candidates"][label] = {
            "family": "feature distance (published ResNet50, 2048-d pooled features, 600 training images)",
            "license": "the app's own model (MIT); no extra download",
            "bytes_extra": 0,
            "metrics": evaluate(scores, meta, groups),
        }
        print(f"{label:<32} {summary(doc['candidates'][label]['metrics'])}", flush=True)

    for name in args.candidate or list(CANDIDATES):
        encoder, text, scale = load_candidate(name)
        emb_dir = WORK / "emb" / name
        emb_dir.mkdir(parents=True, exist_ok=True)
        emb = {}
        for s in sets:
            p = emb_dir / f"{s}.npy"
            if not p.exists():
                np.save(p, embed(encoder, data[s]["seen"]))
            emb[s] = np.load(p)
        t_id, t_rej = text(prompts["in_domain"]), text(prompts["reject"])
        tr_labels = [m["label"] for m in meta["train_sample"]]
        protos = np.stack([emb["train_sample"][[lab == c for lab in tr_labels]].mean(axis=0) for c in classes])
        protos /= np.linalg.norm(protos, axis=1, keepdims=True)
        entry = {
            "family": "zero-shot image-text gate",
            "repo": CANDIDATES[name]["repo"],
            "license": CANDIDATES[name]["license"],
            "logit_scale": scale,
            "params_image_encoder": int(sum(p.numel() for p in encoder.inner.vision_model.parameters()))
            if hasattr(encoder.inner, "vision_model")
            else int(sum(p.numel() for p in encoder.inner.visual.parameters())),
        }
        variants, entry["fused_stats"] = make_scorers(emb, t_id, t_rej, scale, protos)
        for variant, fn in variants.items():
            scores = {s: fn(emb[s]) for s in sets if s != "train_sample"}
            scores["test_all"] = np.concatenate([scores["test_internal"], scores["test_external"]])
            entry[variant] = evaluate(scores, meta, groups)
            print(f"{name + ' ' + variant:<32} {summary(entry[variant])}", flush=True)
        if not args.no_onnx:
            ref = data["samples"]["seen"]
            entry["onnx"] = onnx_measurements(name, encoder, ref, emb["samples"])
            print(
                f"{'':<32} "
                + ", ".join(
                    f"{k}: {v['bytes'] / 2**20:.1f} MiB {v['cpu_1t_median_ms']:.0f} ms (1t) "
                    f"cos>={v['cosine_vs_pytorch_min']:.4f}"
                    for k, v in entry["onnx"].items()
                ),
                flush=True,
            )
        doc["candidates"][name] = entry
        out_json.write_text(json.dumps(doc, indent=2) + "\n", "utf-8", newline="\n")
    out_json.write_text(json.dumps(doc, indent=2) + "\n", "utf-8", newline="\n")
    return 0


def summary(m: dict) -> str:
    return (
        f"t={m['threshold']:.4f} FRR val {m['frr_val']['overall']['rate']:.3f} "
        f"test {m['frr_test_all']['rate']:.3f} | FAR tune {m['far_ood_tune']['overall']['rate']:.3f} "
        f"(near {m['far_ood_tune']['near']['rate']:.3f}, far {m['far_ood_tune']['far']['rate']:.3f}) "
        f"test {m['far_ood_test']['overall']['rate']:.3f} (near {m['far_ood_test']['near']['rate']:.3f}) "
        f"AUROC tune {m['auroc_val_vs_ood_tune']:.3f} near {m['auroc_near_tune']:.3f}"
    )


if __name__ == "__main__":
    sys.exit(main())
