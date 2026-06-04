#!/usr/bin/env python
"""Export a small causal-LM inference trace for the static visualizer.

Install dependencies in your own Python environment:

    pip install torch transformers

Example:

    python scripts/export_trace.py \
      --model gpt2 \
      --prompt "The capital of France is" \
      --out public/traces/gpt2-capital.json

The script records public forward-pass tensors that Hugging Face exposes:
token ids, attention weights, hidden-state statistics, top logits, and
past_key_values shapes. It does not expose hidden chain-of-thought.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


COLORS = [
    "#2f6f73",
    "#9a5b2f",
    "#5d6fa8",
    "#2e7d55",
    "#a94b43",
    "#b07912",
    "#59636a",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a causal-LM trace JSON.")
    parser.add_argument("--model", default="gpt2", help="Hugging Face model id")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--steps", type=int, default=2, help="Greedy decode steps")
    parser.add_argument("--top-k", type=int, default=8, help="Top logits to keep")
    parser.add_argument(
        "--layers",
        default="first,middle,last",
        help="Comma-separated layer ids, or first,middle,last",
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, mps, ...")
    return parser.parse_args()


def tensor_stats(tensor: Any) -> dict[str, float]:
    tensor = tensor.detach().float().cpu()
    return {
        "min": round(float(tensor.min()), 6),
        "max": round(float(tensor.max()), 6),
        "mean": round(float(tensor.mean()), 6),
        "std": round(float(tensor.std(unbiased=False)), 6),
        "l2": round(float(tensor.norm()), 6),
    }


def shape_of(tensor: Any) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def decode_one(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    if text == "":
        return tokenizer.convert_ids_to_tokens([token_id])[0]
    return text


def role_token(
    tokenizer: Any,
    token_id: int,
    position: int,
    role: str,
) -> dict[str, Any]:
    text = decode_one(tokenizer, token_id)
    return {
        "id": int(token_id),
        "text": text,
        "position": int(position),
        "role": role,
        "bytes": len(text.encode("utf-8")),
        "vectorNorm": 0,
        "color": COLORS[position % len(COLORS)],
    }


def choose_layers(layer_count: int, spec: str) -> list[int]:
    if spec == "first,middle,last":
        return sorted(set([0, layer_count // 2, layer_count - 1]))
    chosen = [int(item.strip()) for item in spec.split(",") if item.strip()]
    return [layer for layer in chosen if 0 <= layer < layer_count]


def cache_trace(past_key_values: Any, layer_index: int) -> dict[str, Any]:
    key, value = past_key_values[layer_index][:2]
    sequence_length = int(key.shape[-2])
    memory_mb = (key.numel() + value.numel()) * key.element_size() / (1024 * 1024)
    return {
        "sequenceLength": sequence_length,
        "keyShape": shape_of(key),
        "valueShape": shape_of(value),
        "memoryMB": round(memory_mb, 4),
        "update": "past_key_values stores historical K/V for this layer; decode appends one position per generated token.",
    }


def normalize_rows(matrix: Any) -> list[list[float]]:
    rows = matrix.detach().float().cpu().tolist()
    return [[round(float(value), 6) for value in row] for row in rows]


def top_logits(tokenizer: Any, logits: Any, top_k: int) -> list[dict[str, Any]]:
    import torch

    probs = torch.softmax(logits.float(), dim=-1)
    values, ids = torch.topk(probs, k=top_k)
    result = []
    for probability, token_id in zip(values.tolist(), ids.tolist()):
        logit = logits[int(token_id)].detach().float().cpu().item()
        result.append(
            {
                "token": decode_one(tokenizer, int(token_id)),
                "id": int(token_id),
                "logit": round(float(logit), 6),
                "probability": round(float(probability), 8),
                "note": "model forward value",
            }
        )
    return result


def model_profile_id(model_type: str) -> str:
    lowered = model_type.lower()
    if "llama" in lowered:
        return "llama-style"
    if "gpt2" in lowered:
        return "gpt2-small"
    return "gpt-style"


def default_stages(is_llama: bool) -> list[dict[str, str]]:
    position = "RoPE rotary position on Q/K" if is_llama else "learned/absolute position"
    norm = "RMSNorm + residual" if is_llama else "LayerNorm / residual"
    mlp = "SwiGLU gated MLP" if is_llama else "feed-forward MLP"
    return [
        {
            "id": "tokenize",
            "title": "Tokenize",
            "subtitle": "text -> token ids",
            "description": "Tokenizer converts the string into vocabulary ids consumed by the model.",
            "formula": "ids = tokenizer(text)",
            "shape": "[batch, seq]",
        },
        {
            "id": "embedding",
            "title": "Embedding",
            "subtitle": "ids -> vectors",
            "description": "Token ids are mapped to hidden-size vectors.",
            "formula": "x0 = W_E[ids]",
            "shape": "[batch, seq, d_model]",
        },
        {
            "id": "position",
            "title": "Position",
            "subtitle": position,
            "description": "Position information enters the attention computation according to the model family.",
            "formula": "x = position(x)",
            "shape": "[seq, d_model] or [heads, seq, head_dim]",
        },
        {
            "id": "qkv",
            "title": "Q/K/V",
            "subtitle": "linear projections",
            "description": "Each block projects residual stream values into query, key, and value tensors.",
            "formula": "Q,K,V = xW_q, xW_k, xW_v",
            "shape": "[batch, heads, seq, head_dim]",
        },
        {
            "id": "attention",
            "title": "Causal Attention",
            "subtitle": "masked weighted read",
            "description": "A causal mask prevents tokens from reading future positions.",
            "formula": "softmax(QK^T / sqrt(d) + mask)V",
            "shape": "[heads, target, source]",
        },
        {
            "id": "residual",
            "title": "Residual Stream",
            "subtitle": norm,
            "description": "Block outputs are written back to the residual stream.",
            "formula": "x <- x + block(norm(x))",
            "shape": "[batch, seq, d_model]",
        },
        {
            "id": "mlp",
            "title": "MLP",
            "subtitle": mlp,
            "description": "Per-position nonlinear transformation after attention.",
            "formula": "mlp(x)",
            "shape": "[batch, seq, d_ff]",
        },
        {
            "id": "logits",
            "title": "LM Head",
            "subtitle": "hidden -> vocab scores",
            "description": "The final hidden state is projected to vocabulary logits.",
            "formula": "logits = h_last W_U",
            "shape": "[batch, vocab]",
        },
        {
            "id": "sampling",
            "title": "Selection",
            "subtitle": "argmax or sampling",
            "description": "A decoding strategy selects the next token from the probability distribution.",
            "formula": "p = softmax(logits / T)",
            "shape": "[top_k]",
        },
        {
            "id": "kvcache",
            "title": "KV Cache",
            "subtitle": "reuse history",
            "description": "Past keys and values are cached per layer for fast autoregressive decoding.",
            "formula": "cache = concat(cache, K_new, V_new)",
            "shape": "[layers, heads, seq, head_dim]",
        },
    ]


def comparisons() -> list[dict[str, str]]:
    return [
        {
            "axis": "Position",
            "gpt": "GPT-2 style commonly uses learned absolute position embedding.",
            "llama": "LLaMA style uses RoPE on Q/K.",
        },
        {
            "axis": "Norm",
            "gpt": "GPT-family public models vary; GPT-2 uses LayerNorm.",
            "llama": "LLaMA-family typically uses RMSNorm.",
        },
        {
            "axis": "MLP",
            "gpt": "GPT-2 uses a GELU feed-forward MLP.",
            "llama": "LLaMA-family uses SwiGLU.",
        },
        {
            "axis": "KV cache",
            "gpt": "Classic MHA has equal Q/K/V head counts.",
            "llama": "GQA/MQA variants reduce KV heads.",
        },
    ]


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Run: pip install torch transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)
    layer_count = int(getattr(model.config, "num_hidden_layers", 0))
    hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0))
    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))
    model_type = str(getattr(model.config, "model_type", args.model))
    selected_layers = choose_layers(layer_count, args.layers)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=True,
        )

    token_ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
    tokens = [role_token(tokenizer, token_id, index, "prompt") for index, token_id in enumerate(token_ids)]
    for index, token in enumerate(tokens):
        hidden = outputs.hidden_states[0][0, index]
        token["vectorNorm"] = round(float(hidden.detach().float().cpu().norm()), 6)

    decode_steps = []
    past = outputs.past_key_values
    next_logits = outputs.logits[0, -1]
    generated_text = ""

    for step_index in range(args.steps):
        top = top_logits(tokenizer, next_logits, args.top_k)
        next_id = int(top[0]["id"])
        output_position = len(tokens)
        output_token = role_token(tokenizer, next_id, output_position, "generated")
        generated_text += output_token["text"]
        tokens.append(output_token)
        decode_steps.append(
            {
                "index": step_index,
                "phase": "prefill" if step_index == 0 else "decode",
                "inputPosition": output_position - 1,
                "outputToken": output_token,
                "selectedTokenId": next_id,
                "entropy": round(float(-(torch.softmax(next_logits.float(), dim=-1) * torch.log_softmax(next_logits.float(), dim=-1)).sum()), 6),
                "temperature": 1,
                "topP": 1,
                "topLogits": top,
                "kvCache": cache_trace(past, selected_layers[-1]),
            }
        )
        with torch.no_grad():
            step_outputs = model(
                input_ids=torch.tensor([[next_id]], device=args.device),
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=True,
            )
        past = step_outputs.past_key_values
        next_logits = step_outputs.logits[0, -1]

    layers = []
    attentions = outputs.attentions or []
    hidden_states = outputs.hidden_states or []
    for layer_index in selected_layers:
        attention = []
        if attentions:
            attention = normalize_rows(attentions[layer_index][0, 0])
        hidden = hidden_states[min(layer_index + 1, len(hidden_states) - 1)][0]
        previous = hidden_states[max(layer_index, 0)][0]
        delta = hidden - previous
        key, _value = outputs.past_key_values[layer_index][:2]
        heads = int(key.shape[1])
        head_dim = int(key.shape[-1])
        layers.append(
            {
                "index": layer_index,
                "label": f"layer {layer_index}",
                "attentionHead": 0,
                "heads": heads,
                "kvHeads": heads,
                "attention": attention,
                "qkScale": round(1 / math.sqrt(head_dim), 6),
                "hidden": tensor_stats(hidden),
                "residual": tensor_stats(previous),
                "mlp": tensor_stats(delta),
                "kvCache": cache_trace(outputs.past_key_values, layer_index),
                "probes": [
                    {
                        "name": "last hidden slice",
                        "shape": [8],
                        "sample": [
                            round(float(value), 6)
                            for value in hidden[-1, :8].detach().float().cpu().tolist()
                        ],
                    },
                    {
                        "name": "attention row for last prompt token",
                        "shape": [len(token_ids)],
                        "sample": attention[-1] if attention else [],
                    },
                ],
            }
        )

    trace = {
        "id": Path(args.out).stem,
        "title": f"{args.model}: exported trace",
        "prompt": args.prompt,
        "generatedText": generated_text,
        "modelProfileId": model_profile_id(model_type),
        "family": model_type,
        "parameterScale": str(getattr(model.config, "architectures", ["causal LM"])[0]),
        "layerCount": layer_count,
        "hiddenSize": hidden_size,
        "vocabSize": vocab_size,
        "tokenizer": tokenizer.__class__.__name__,
        "source": {
            "type": "hf-export",
            "label": "Hugging Face forward trace",
            "modelId": args.model,
            "command": " ".join(
                [
                    "python scripts/export_trace.py",
                    f"--model {args.model}",
                    f"--prompt {json.dumps(args.prompt, ensure_ascii=False)}",
                    f"--out {args.out}",
                ]
            ),
            "note": "Values were exported from a real model forward pass with output_attentions, output_hidden_states, and use_cache enabled.",
        },
        "tensors": [
            {"label": "input_ids", "shape": shape_of(input_ids), "dtype": "int64"},
            {
                "label": "hidden_states",
                "shape": shape_of(outputs.hidden_states[-1]),
                "dtype": "float32",
            },
            {
                "label": "next_logits",
                "shape": shape_of(outputs.logits[:, -1, :]),
                "dtype": "float32",
            },
        ],
        "stages": default_stages("llama" in model_type.lower()),
        "tokens": tokens,
        "layers": layers,
        "decodeSteps": decode_steps,
        "comparisons": comparisons(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
