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


def first_tensor(value: Any) -> Any | None:
    if hasattr(value, "detach") and hasattr(value, "shape"):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def tensor_sample(tensor: Any, limit: int = 8) -> list[float]:
    tensor = tensor.detach().float().cpu()
    try:
        if tensor.ndim == 0:
            selected = tensor.reshape(1)
        elif tensor.ndim == 1:
            selected = tensor
        elif tensor.ndim == 2:
            selected = tensor[-1]
        elif tensor.ndim == 3:
            selected = tensor[0, -1]
        elif tensor.ndim == 4:
            selected = tensor[0, 0, -1]
        else:
            selected = tensor.reshape(-1)
    except IndexError:
        selected = tensor.reshape(-1)
    return [round(float(value), 6) for value in selected.reshape(-1)[:limit].tolist()]


def tensor_record(value: Any) -> dict[str, Any] | None:
    tensor = first_tensor(value)
    if tensor is None:
        return None
    return {
        "shape": shape_of(tensor),
        "sample": tensor_sample(tensor),
        "stats": tensor_stats(tensor),
    }


def module_at(root: Any, path: str) -> Any | None:
    current = root
    for part in path.split("."):
        try:
            current = current[int(part)] if part.isdigit() else getattr(current, part)
        except (AttributeError, IndexError, TypeError, KeyError):
            return None
    return current


def operator_spec(
    operator_id: str,
    name: str,
    group: str,
    stage_id: str,
    path: str,
    expression: str,
    latex: str,
    description: str,
) -> dict[str, str]:
    return {
        "id": operator_id,
        "name": name,
        "group": group,
        "stageId": stage_id,
        "path": path,
        "expression": expression,
        "latex": latex,
        "description": description,
    }


def hook_specs_for_layer(layer_index: int, is_llama: bool) -> list[dict[str, str]]:
    if is_llama:
        prefix = f"model.layers.{layer_index}"
        return [
            operator_spec(
                "rms-attn",
                "RMSNorm before Attention",
                "attention",
                "residual",
                f"{prefix}.input_layernorm",
                "a0 = RMSNorm(x0)",
                "\\hat{X}_0=\\operatorname{RMSNorm}(X_0)",
                "Measured module input/output for the pre-attention RMSNorm.",
            ),
            operator_spec(
                "q-proj",
                "Query Projection",
                "attention",
                "qkv",
                f"{prefix}.self_attn.q_proj",
                "Q = a0 W_q",
                "Q=\\hat{X}_0W_q",
                "Measured query projection output.",
            ),
            operator_spec(
                "k-proj",
                "Key Projection",
                "attention",
                "qkv",
                f"{prefix}.self_attn.k_proj",
                "K = a0 W_k",
                "K=\\hat{X}_0W_k",
                "Measured key projection output before model-internal positional handling.",
            ),
            operator_spec(
                "v-proj",
                "Value Projection",
                "attention",
                "qkv",
                f"{prefix}.self_attn.v_proj",
                "V = a0 W_v",
                "V=\\hat{X}_0W_v",
                "Measured value projection output.",
            ),
            operator_spec(
                "attn-out",
                "Attention Output Projection",
                "attention",
                "attention",
                f"{prefix}.self_attn.o_proj",
                "O_attn = concat(ctx) W_o",
                "O_{\\mathrm{attn}}=\\operatorname{Concat}(C_1,\\ldots,C_h)W_o",
                "Measured attention output projection input/output.",
            ),
            operator_spec(
                "rms-mlp",
                "RMSNorm before MLP",
                "mlp",
                "residual",
                f"{prefix}.post_attention_layernorm",
                "m0 = RMSNorm(x1)",
                "\\hat{X}_1=\\operatorname{RMSNorm}(X_1)",
                "Measured module input/output for the pre-MLP RMSNorm.",
            ),
            operator_spec(
                "gate-proj",
                "SwiGLU Gate Projection",
                "mlp",
                "mlp",
                f"{prefix}.mlp.gate_proj",
                "G = x W_gate",
                "G=\\hat{X}_1W_{\\mathrm{gate}}",
                "Measured gate branch projection.",
            ),
            operator_spec(
                "up-proj",
                "SwiGLU Up Projection",
                "mlp",
                "mlp",
                f"{prefix}.mlp.up_proj",
                "U = x W_up",
                "U=\\hat{X}_1W_{\\mathrm{up}}",
                "Measured up branch projection.",
            ),
            operator_spec(
                "down-proj",
                "MLP Down Projection",
                "mlp",
                "mlp",
                f"{prefix}.mlp.down_proj",
                "O_mlp = SiLU(G) * U * W_down",
                "O_{\\mathrm{mlp}}=(\\operatorname{SiLU}(G)\\odot U)W_{\\mathrm{down}}",
                "Measured down projection input/output after the gated activation.",
            ),
        ]

    prefix = f"transformer.h.{layer_index}"
    return [
        operator_spec(
            "ln-attn",
            "LayerNorm before Attention",
            "attention",
            "residual",
            f"{prefix}.ln_1",
            "a0 = LayerNorm(x0)",
            "\\hat{X}_0=\\operatorname{LayerNorm}(X_0)",
            "Measured module input/output for the pre-attention LayerNorm.",
        ),
        operator_spec(
            "qkv-proj",
            "Q/K/V Linear Projection",
            "attention",
            "qkv",
            f"{prefix}.attn.c_attn",
            "Q,K,V = a0 W_qkv",
            "[Q,K,V]=\\hat{X}_0W_{qkv}",
            "Measured fused Q/K/V projection output.",
        ),
        operator_spec(
            "attn-out",
            "Attention Output Projection",
            "attention",
            "attention",
            f"{prefix}.attn.c_proj",
            "O_attn = concat(ctx) W_o",
            "O_{\\mathrm{attn}}=\\operatorname{Concat}(C_1,\\ldots,C_h)W_o",
            "Measured attention output projection input/output.",
        ),
        operator_spec(
            "ln-mlp",
            "LayerNorm before MLP",
            "mlp",
            "residual",
            f"{prefix}.ln_2",
            "m0 = LayerNorm(x1)",
            "\\hat{X}_1=\\operatorname{LayerNorm}(X_1)",
            "Measured module input/output for the pre-MLP LayerNorm.",
        ),
        operator_spec(
            "mlp-up",
            "MLP Up Projection",
            "mlp",
            "mlp",
            f"{prefix}.mlp.c_fc",
            "U = m0 W_in + b_in",
            "U=\\hat{X}_1W_{\\mathrm{in}}+b_{\\mathrm{in}}",
            "Measured MLP expansion projection.",
        ),
        operator_spec(
            "mlp-act",
            "GELU Activation",
            "mlp",
            "mlp",
            f"{prefix}.mlp.act",
            "A = GELU(U)",
            "A=\\operatorname{GELU}(U)",
            "Measured activation input/output when the activation is exposed as a module.",
        ),
        operator_spec(
            "mlp-down",
            "MLP Down Projection",
            "mlp",
            "mlp",
            f"{prefix}.mlp.c_proj",
            "O_mlp = A W_out + b_out",
            "O_{\\mathrm{mlp}}=AW_{\\mathrm{out}}+b_{\\mathrm{out}}",
            "Measured MLP down projection.",
        ),
    ]


def register_operator_hooks(
    model: Any,
    selected_layers: list[int],
    is_llama: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], list[Any]]:
    recordings: dict[str, dict[str, Any]] = {}
    specs_by_key: dict[str, dict[str, str]] = {}
    handles = []

    for layer_index in selected_layers:
        for spec in hook_specs_for_layer(layer_index, is_llama):
            module = module_at(model, spec["path"])
            if module is None or not hasattr(module, "register_forward_hook"):
                continue
            key = f"{layer_index}:{spec['id']}"
            specs_by_key[key] = spec

            def capture(_module: Any, inputs: Any, output: Any, hook_key: str = key) -> None:
                if hook_key in recordings:
                    return
                recordings[hook_key] = {
                    "input": tensor_record(inputs),
                    "output": tensor_record(output),
                }

            handles.append(module.register_forward_hook(capture))

    return recordings, specs_by_key, handles


def source_record(source_type: str, label: str, measured: bool, note: str) -> dict[str, Any]:
    return {
        "type": source_type,
        "label": label,
        "measured": measured,
        "note": note,
    }


def operator_from_hook(
    layer_index: int,
    spec: dict[str, str],
    record: dict[str, Any],
) -> dict[str, Any] | None:
    input_record = record.get("input")
    output_record = record.get("output")
    if not input_record or not output_record:
        return None

    return {
        "id": spec["id"],
        "group": spec["group"],
        "stageId": spec["stageId"],
        "name": spec["name"],
        "expression": spec["expression"],
        "latex": spec["latex"],
        "inputTensor": f"L{layer_index}.{spec['id']}.input",
        "outputTensor": f"L{layer_index}.{spec['id']}.output",
        "inputShape": input_record["shape"],
        "outputShape": output_record["shape"],
        "inputSample": input_record["sample"],
        "outputSample": output_record["sample"],
        "sample": output_record["sample"],
        "inputStats": input_record["stats"],
        "outputStats": output_record["stats"],
        "reads": [f"L{layer_index}.{spec['id']}.input"],
        "writes": [f"L{layer_index}.{spec['id']}.output"],
        "source": source_record(
            "hf-hook",
            "HF module hook",
            True,
            f"Captured from {spec['path']} during the prefill forward pass.",
        ),
        "description": spec["description"],
        "debugNote": "This operator input/output was captured from a real module forward hook.",
    }


def measured_operator(
    operator_id: str,
    group: str,
    stage_id: str,
    name: str,
    expression: str,
    latex: str,
    input_shape: list[int],
    output_shape: list[int],
    input_sample: list[float],
    output_sample: list[float],
    description: str,
    note: str,
) -> dict[str, Any]:
    return {
        "id": operator_id,
        "group": group,
        "stageId": stage_id,
        "name": name,
        "expression": expression,
        "latex": latex,
        "inputTensor": f"{operator_id}.input",
        "outputTensor": f"{operator_id}.output",
        "inputShape": input_shape,
        "outputShape": output_shape,
        "inputSample": input_sample,
        "outputSample": output_sample,
        "sample": output_sample,
        "reads": [f"{operator_id}.input"],
        "writes": [f"{operator_id}.output"],
        "source": source_record("hf-forward", "HF forward output", True, note),
        "description": description,
        "debugNote": note,
    }


def build_exported_operators(
    layer_index: int,
    token_ids: list[int],
    previous_hidden: Any,
    hidden: Any,
    attention: list[list[float]],
    key: Any,
    value: Any,
    recordings: dict[str, dict[str, Any]],
    specs_by_key: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    operators: list[dict[str, Any]] = []

    if layer_index == 0:
        operators.append(
            {
                "id": "token-embedding",
                "group": "embedding",
                "stageId": "embedding",
                "name": "Token / Position Embedding Output",
                "expression": "X_0 = embedding(input_ids)",
                "latex": "X_0=\\operatorname{Embedding}(\\mathrm{input\\_ids})",
                "inputTensor": "input_ids",
                "outputTensor": "hidden_states[0]",
                "inputShape": [1, len(token_ids)],
                "outputShape": shape_of(previous_hidden),
                "inputSample": [float(value) for value in token_ids[:8]],
                "outputSample": tensor_sample(previous_hidden),
                "sample": tensor_sample(previous_hidden),
                "inputStats": None,
                "outputStats": tensor_stats(previous_hidden),
                "reads": ["input_ids"],
                "writes": ["hidden_states[0]"],
                "source": source_record(
                    "hf-forward",
                    "HF hidden_states",
                    True,
                    "Captured from outputs.hidden_states[0] during the real prefill forward pass.",
                ),
                "description": "Tokenizer ids after the model embedding stack becomes the initial residual stream.",
                "debugNote": "This is the measured initial hidden state exposed by Hugging Face hidden_states.",
            }
        )

    for key_name, spec in specs_by_key.items():
        key_layer, _operator_id = key_name.split(":", 1)
        if int(key_layer) != layer_index:
            continue
        operator = operator_from_hook(layer_index, spec, recordings.get(key_name, {}))
        if operator:
            operators.append(operator)

    if attention:
        attention_shape = [1, int(key.shape[1]), len(attention), len(attention[0])]
        attention_sample = attention[-1][:8]
        operators.append(
            measured_operator(
                "attn-softmax",
                "attention",
                "attention",
                "Attention Probabilities",
                "A = softmax(masked_scores)",
                "A=\\operatorname{softmax}(S+M)",
                attention_shape,
                attention_shape,
                attention_sample,
                attention_sample,
                "Measured attention probabilities exposed by output_attentions.",
                "The raw pre-softmax score tensor is not exposed by every model, so this trace records the measured probabilities.",
            )
        )

    key_sample = tensor_sample(key)
    value_sample = tensor_sample(value)
    operators.append(
        measured_operator(
            "kv-write",
            "cache",
            "kvcache",
            "KV Cache Write",
            "cache_l = append(K,V)",
            "\\mathrm{Cache}_{\\ell}\\leftarrow\\operatorname{append}(K,V)",
            [2, *shape_of(key)],
            [2, *shape_of(value)],
            key_sample,
            value_sample,
            "Measured K/V tensors stored in past_key_values for this layer.",
            "Captured from outputs.past_key_values after the prefill forward pass.",
        )
    )

    operators.append(
        measured_operator(
            "layer-output",
            "mlp",
            "residual",
            "Layer Output Hidden State",
            "X_{l+1} = block(X_l)",
            "X_{\\ell+1}=\\operatorname{Block}_{\\ell}(X_{\\ell})",
            shape_of(previous_hidden),
            shape_of(hidden),
            tensor_sample(previous_hidden),
            tensor_sample(hidden),
            "Measured hidden state after this decoder block.",
            "Captured from outputs.hidden_states[layer + 1].",
        )
    )

    return operators


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
    if not selected_layers:
        raise SystemExit("No valid layers selected for this model.")
    is_llama = "llama" in model_type.lower()

    recordings, specs_by_key, hook_handles = register_operator_hooks(
        model,
        selected_layers,
        is_llama,
    )
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=True,
            )
    finally:
        for handle in hook_handles:
            handle.remove()

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
        key, value = outputs.past_key_values[layer_index][:2]
        heads = int(getattr(model.config, "num_attention_heads", key.shape[1]))
        kv_heads = int(key.shape[1])
        head_dim = int(key.shape[-1])
        layers.append(
            {
                "index": layer_index,
                "label": f"layer {layer_index}",
                "attentionHead": 0,
                "heads": heads,
                "kvHeads": kv_heads,
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
                "operators": build_exported_operators(
                    layer_index,
                    token_ids,
                    previous,
                    hidden,
                    attention,
                    key,
                    value,
                    recordings,
                    specs_by_key,
                ),
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
        "stages": default_stages(is_llama),
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
