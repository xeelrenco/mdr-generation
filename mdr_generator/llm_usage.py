"""Track LLM token usage and estimate pipeline costs."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import cfg_float
from .utils import save_json


@dataclass
class LlmCallRecord:
    provider: str
    model: str
    stage: str
    call_type: str  # pdf | text
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LlmCostLine:
    stage: str
    provider: str
    model: str
    call_type: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LlmUsageSummary:
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    lines: List[LlmCostLine] = field(default_factory=list)
    provider_cost_usd: Dict[str, float] = field(default_factory=dict)
    pricing_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_calls": self.total_calls,
            "provider_cost_usd": {
                k: round(v, 4) for k, v in sorted(self.provider_cost_usd.items())
            },
            "pricing_note": self.pricing_note,
            "lines": [line.to_dict() for line in self.lines],
        }


# USD per 1M tokens — stime default, sovrascrivibili in settings.toml ([llm_pricing.*])
_DEFAULT_PRICE_PER_1M: Dict[Tuple[str, str], float] = {
    ("gpt-5.5", "input"): 5.00,
    ("gpt-5.5", "output"): 30.00,
    ("gpt-4o", "input"): 2.50,
    ("gpt-4o", "output"): 10.00,
    ("gpt-4o-mini", "input"): 0.15,
    ("gpt-4o-mini", "output"): 0.60,
    ("gemini-2.5-pro", "input"): 1.25,
    ("gemini-2.5-pro", "output"): 5.00,
    ("gemini-2.5-flash", "input"): 0.075,
    ("gemini-2.5-flash", "output"): 0.30,
    ("claude-sonnet-4-6", "input"): 3.00,
    ("claude-sonnet-4-6", "output"): 15.00,
    ("claude-opus-4-8", "input"): 5.00,
    ("claude-opus-4-8", "output"): 25.00,
    ("claude-haiku-4-5", "input"): 0.80,
    ("claude-haiku-4-5", "output"): 4.00,
}


class LlmUsageTracker:
    def __init__(self) -> None:
        self.calls: List[LlmCallRecord] = []

    def reset(self) -> None:
        self.calls.clear()

    def record(
        self,
        provider: str,
        model: str,
        stage: str,
        call_type: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.calls.append(
            LlmCallRecord(
                provider=provider,
                model=model,
                stage=stage or "unknown",
                call_type=call_type,
                input_tokens=max(0, int(input_tokens or 0)),
                output_tokens=max(0, int(output_tokens or 0)),
            )
        )


_tracker = LlmUsageTracker()
_usage_lock = threading.Lock()


def reset_usage_tracker() -> None:
    with _usage_lock:
        _tracker.reset()


def record_llm_usage(
    provider: str,
    model: str,
    stage: str,
    call_type: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    with _usage_lock:
        _tracker.record(provider, model, stage, call_type, input_tokens, output_tokens)


def _model_price_per_1m(model: str, direction: str) -> float:
    key = model.lower().strip()
    default = _DEFAULT_PRICE_PER_1M.get((key, direction), 0.0)
    safe = key.replace(".", "_").replace("-", "_")
    cfg_key = f"LLM_PRICE_USD_PER_1M_{direction.upper()}_{safe}"
    price = cfg_float(cfg_key, default)
    if price > 0:
        return price
    # fallback: longest prefix match in defaults
    for (name, dir_name), val in _DEFAULT_PRICE_PER_1M.items():
        if dir_name == direction and key.startswith(name.split("-")[0]):
            return val
    return default


def _estimate_call_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price = _model_price_per_1m(model, "input")
    out_price = _model_price_per_1m(model, "output")
    if in_price <= 0 and out_price <= 0:
        return 0.0
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


def build_usage_summary() -> LlmUsageSummary:
    grouped: Dict[Tuple[str, str, str, str], LlmCostLine] = {}
    total_in = 0
    total_out = 0

    for call in _tracker.calls:
        total_in += call.input_tokens
        total_out += call.output_tokens
        key = (call.stage, call.provider, call.model, call.call_type)
        if key not in grouped:
            grouped[key] = LlmCostLine(
                stage=call.stage,
                provider=call.provider,
                model=call.model,
                call_type=call.call_type,
                calls=0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
            )
        line = grouped[key]
        line.calls += 1
        line.input_tokens += call.input_tokens
        line.output_tokens += call.output_tokens
        line.cost_usd += _estimate_call_cost_usd(
            call.model, call.input_tokens, call.output_tokens
        )

    lines = sorted(
        grouped.values(),
        key=lambda x: (-x.cost_usd, x.stage, x.model),
    )
    total_cost = sum(line.cost_usd for line in lines)
    provider_cost: Dict[str, float] = {}
    for line in lines:
        provider_cost[line.provider] = provider_cost.get(line.provider, 0.0) + line.cost_usd
    return LlmUsageSummary(
        total_cost_usd=round(total_cost, 4),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_calls=len(_tracker.calls),
        lines=lines,
        provider_cost_usd=provider_cost,
        pricing_note=(
            "Stima da token API e tariffe default/settings.toml ([llm_pricing.*]). "
            "gpt-5.5: $5/M input, $30/M output (standard ≤272K). "
            "OpenAI e Gemini/Vertex fatturati separatamente. "
            "Cached input e PDF multimodali possono divergere dal billing reale."
        ),
    )


def format_cost_usd(amount: float) -> str:
    if amount <= 0:
        return "$0.00"
    if amount < 0.01:
        return f"${amount:.4f}"
    return f"${amount:.2f}"


def format_token_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def provider_billing_label(provider: str) -> str:
    labels = {
        "openai": "OpenAI",
        "gemini": "Gemini/Vertex",
        "claude": "Anthropic",
    }
    return labels.get(provider, provider)


def format_usage_console(summary: LlmUsageSummary) -> str:
    if summary.total_calls == 0:
        return "Stima costi LLM: nessuna chiamata registrata"
    head = (
        f"Stima costi LLM: {format_cost_usd(summary.total_cost_usd)} USD "
        f"({format_token_count(summary.total_input_tokens)} input + "
        f"{format_token_count(summary.total_output_tokens)} output token, "
        f"{summary.total_calls} chiamate)"
    )
    if not summary.lines:
        return head
    parts = [head]
    for provider, amount in sorted(
        summary.provider_cost_usd.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        parts.append(
            f"  {provider_billing_label(provider)}: {format_cost_usd(amount)}"
        )
    for line in summary.lines:
        parts.append(
            f"  {line.stage} ({line.model}, {line.call_type}): "
            f"{format_cost_usd(line.cost_usd)} — {line.calls} chiamate"
        )
    return "\n".join(parts)


def save_usage_audit(json_dir, summary: LlmUsageSummary) -> None:
    from pathlib import Path

    path = Path(json_dir) / "llm_usage_audit.json"
    save_json(path, summary.to_dict())
