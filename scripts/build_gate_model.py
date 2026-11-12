"""Build the input check's image encoder for the browser: a Keras copy of a Hugging Face CLIP vision tower.

The browser already runs TensorFlow.js, so the gate's image encoder is rebuilt in Keras with the original
weights and converted with the same TF.js converter as the main model (scripts/convert_to_tfjs.sh). No
second runtime is needed. Three steps, each in the venv that has its framework:

1. ``export`` (gate venv, PyTorch): writes the vision weights, the config, the text embeddings of the prompts
   and PyTorch reference embeddings of the shipped sample photos to build/gate/<name>/.
2. ``keras`` (TensorFlow venv): rebuilds the vision tower in Keras, loads the weights, checks the embeddings
   against PyTorch (cosine >= 0.9999 on every sample), and exports a SavedModel to build/gate/<name>/saved_model.
3. ``scripts/convert_to_tfjs.sh --gate <name>`` converts the SavedModel (float16 and uint8 weights).

Signature ``serving_default``:
    input   image      float32 [N, 224, 224, 3]  the 224 x 224 RGB image the classifier sees, 0..255
    output  embedding  float32 [N, D]            unit-length image embedding

Usage:
    .work/venv-gate/Scripts/python scripts/build_gate_model.py export tinyclip-vit-8m-16-yfcc15m
    .work/venv-tf/Scripts/python   scripts/build_gate_model.py keras  tinyclip-vit-8m-16-yfcc15m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "gate"
WORK = ROOT / ".work" / "gate"
PROMPTS = ROOT / "src" / "poultry_disease" / "data" / "input_check_prompts.json"
MIN_COS = 0.9999
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def export(name: str) -> int:
    import torch

    sys.path.insert(0, str(ROOT / "scripts"))
    from gate_study import CANDIDATES, embed, load_candidate

    if CANDIDATES[name]["kind"] != "hf-clip":
        raise SystemExit(f"{name}: only Hugging Face CLIP vision towers can be rebuilt in Keras here")
    encoder, text, scale = load_candidate(name)
    model = encoder.inner
    out = BUILD / name
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().numpy() for k, v in model.vision_model.state_dict().items()}
    state["visual_projection.weight"] = model.visual_projection.weight.detach().numpy()
    np.savez(out / "weights.npz", **state)
    vc = model.config.vision_config
    config = {
        "name": name,
        "repo": CANDIDATES[name]["repo"],
        "license": CANDIDATES[name]["license"],
        "hidden_size": vc.hidden_size,
        "intermediate_size": vc.intermediate_size,
        "num_hidden_layers": vc.num_hidden_layers,
        "num_attention_heads": vc.num_attention_heads,
        "patch_size": vc.patch_size,
        "image_size": vc.image_size,
        "hidden_act": vc.hidden_act,
        "layer_norm_eps": vc.layer_norm_eps,
        "projection_dim": model.config.projection_dim,
        "mean": CLIP_MEAN,
        "std": CLIP_STD,
        "logit_scale": scale,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", "utf-8")
    prompts = json.loads(PROMPTS.read_text("utf-8"))
    np.savez(
        out / "text.npz",
        in_domain=text(prompts["in_domain"]),
        reject=text(prompts["reject"]),
    )
    seen = np.load(WORK / "samples.npz")["seen"]
    with torch.no_grad():
        np.save(out / "reference_seen.npy", seen)
        np.save(out / "reference_embedding.npy", embed(encoder, seen))
    print(f"exported {name} -> {out}")
    return 0


def build_keras(config: dict, weights: dict):
    import keras
    from keras import layers, ops

    d, heads = config["hidden_size"], config["num_attention_heads"]
    p, size = config["patch_size"], config["image_size"]
    n_patches = (size // p) ** 2
    eps = config["layer_norm_eps"]

    def act(x):
        if config["hidden_act"] == "quick_gelu":
            return x * ops.sigmoid(1.702 * x)
        if config["hidden_act"] == "gelu":  # exact GELU via Erf (TF.js has Erf but not Erfc, which ops.gelu uses)
            return 0.5 * x * (1.0 + ops.erf(x * 0.7071067811865476))
        raise ValueError(config["hidden_act"])

    def ln(name):
        return layers.LayerNormalization(epsilon=eps, name=name)

    image = keras.Input((size, size, 3), dtype="float32", name="image")
    mean = np.asarray(config["mean"], np.float32) * 255.0
    std = np.asarray(config["std"], np.float32) * 255.0
    x = (image - mean) / std
    x = layers.Conv2D(d, p, strides=p, use_bias=False, name="patch_embedding")(x)
    x = layers.Reshape((n_patches, d))(x)

    class Embeddings(layers.Layer):
        def build(self, _):
            self.cls = self.add_weight(shape=(d,), name="class_embedding", initializer="zeros")
            self.pos = self.add_weight(shape=(n_patches + 1, d), name="position_embedding", initializer="zeros")

        def call(self, t):
            n = ops.shape(t)[0]
            cls = ops.broadcast_to(ops.reshape(self.cls, (1, 1, d)), (n, 1, d))
            return ops.concatenate([cls, t], axis=1) + self.pos[None]

    emb = Embeddings(name="embeddings")
    x = emb(x)
    x = ln("pre_layrnorm")(x)
    hd = d // heads
    for i in range(config["num_hidden_layers"]):
        pre = f"encoder.layers.{i}."
        h = ln(pre + "layer_norm1")(x)
        q = layers.Dense(d, name=pre + "q_proj")(h) * (hd**-0.5)
        k = layers.Dense(d, name=pre + "k_proj")(h)
        v = layers.Dense(d, name=pre + "v_proj")(h)

        def split(t):
            return ops.transpose(ops.reshape(t, (-1, n_patches + 1, heads, hd)), (0, 2, 1, 3))

        a = ops.softmax(ops.matmul(split(q), ops.transpose(split(k), (0, 1, 3, 2))), axis=-1)
        o = ops.reshape(ops.transpose(ops.matmul(a, split(v)), (0, 2, 1, 3)), (-1, n_patches + 1, d))
        x = x + layers.Dense(d, name=pre + "out_proj")(o)
        h = ln(pre + "layer_norm2")(x)
        h = layers.Dense(config["intermediate_size"], name=pre + "fc1")(h)
        x = x + layers.Dense(d, name=pre + "fc2")(act(h))
    pooled = ln("post_layernorm")(x[:, 0])
    proj = layers.Dense(config["projection_dim"], use_bias=False, name="visual_projection")(pooled)
    out = layers.Identity(name="embedding")(proj / ops.norm(proj, axis=-1, keepdims=True))
    model = keras.Model(image, {"embedding": out}, name="input_check_image_encoder")

    w = weights
    model.get_layer("patch_embedding").set_weights([w["embeddings.patch_embedding.weight"].transpose(2, 3, 1, 0)])
    emb.set_weights([w["embeddings.class_embedding"], w["embeddings.position_embedding.weight"]])
    for ln_name in ("pre_layrnorm", "post_layernorm"):
        model.get_layer(ln_name).set_weights([w[f"{ln_name}.weight"], w[f"{ln_name}.bias"]])
    for i in range(config["num_hidden_layers"]):
        pre = f"encoder.layers.{i}."
        for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            model.get_layer(pre + proj_name).set_weights(
                [w[f"{pre}self_attn.{proj_name}.weight"].T, w[f"{pre}self_attn.{proj_name}.bias"]]
            )
        for fc in ("fc1", "fc2"):
            model.get_layer(pre + fc).set_weights([w[f"{pre}mlp.{fc}.weight"].T, w[f"{pre}mlp.{fc}.bias"]])
        for ln_name in ("layer_norm1", "layer_norm2"):
            model.get_layer(pre + ln_name).set_weights([w[f"{pre}{ln_name}.weight"], w[f"{pre}{ln_name}.bias"]])
    model.get_layer("visual_projection").set_weights([w["visual_projection.weight"].T])
    return model


def keras_step(name: str) -> int:
    import tensorflow as tf

    out = BUILD / name
    config = json.loads((out / "config.json").read_text("utf-8"))
    weights = dict(np.load(out / "weights.npz"))
    model = build_keras(config, weights)
    seen = np.load(out / "reference_seen.npy").astype(np.float32)
    ref = np.load(out / "reference_embedding.npy")
    got = model.predict(seen, verbose=0)["embedding"]
    cos = (got * ref).sum(axis=1)
    print(f"[check] Keras vs PyTorch embeddings: cosine min {cos.min():.6f} (required >= {MIN_COS})")
    assert cos.min() >= MIN_COS
    saved = out / "saved_model"
    model.export(str(saved), verbose=False)
    sig = tf.saved_model.load(str(saved)).signatures["serving_default"]
    served = sig(image=tf.constant(seen))["embedding"].numpy()
    diff = float(np.abs(served - got).max())
    print(f"[check] SavedModel vs Keras: max |diff| {diff:.2e}")
    assert diff < 1e-5
    print(f"Exported -> {saved}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["export", "keras"])
    ap.add_argument("name")
    args = ap.parse_args()
    return export(args.name) if args.step == "export" else keras_step(args.name)


if __name__ == "__main__":
    sys.exit(main())
