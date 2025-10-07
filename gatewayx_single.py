"""
Gateway X — Single-file Prototype (v30.2)
=========================================

This file consolidates the full prototype into ONE module so you can:
- paste into Cursor,
- run locally with FastAPI/uvicorn,
- quickly iterate before splitting into a multi-file repo.

Endpoints
---------
POST /query
  Body: { "query": str, "budget": int (1..200), "confidence_threshold": float (0.5..0.999) }
  Returns: best claim, confidence, rounds, duels, stop reason, run_id

GET /timeline/{run_id}
  Returns: the full per-round timeline as JSON

GET /timeline/{run_id}/report
  Returns: a small HTML report (no external deps) with sparkline + per-round table

Notes
-----
- By default, USE_REAL_LLM=False (no API calls). Set env GX_USE_REAL_LLM=true and
  ANTHROPIC_API_KEY to use Anthropic. Model defaults to claude-3-5-sonnet-20240620.
- Bootstrap CIs are computed for BTL scores (for CI-separation stopping).
- All initial claims, duels, round summaries are recorded to a transparency ledger.
"""

from __future__ import annotations
import os
import math
import json
import time
import hashlib
import random
import logging
import html
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from itertools import combinations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, conint, confloat

# Optional Anthropic import: only used if GX_USE_REAL_LLM=true
try:
    from anthropic import AsyncAnthropic  # type: ignore
except Exception:  # pragma: no cover
    AsyncAnthropic = None  # type: ignore

# ------------------------------------------------------------------------------
# Config (env-driven; safe defaults for local dev)
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    # Budgets & rounds
    DEFAULT_BUDGET: int = int(os.getenv("GX_DEFAULT_BUDGET", "12"))
    MAX_BUDGET: int = int(os.getenv("GX_MAX_BUDGET", "200"))
    MAX_ROUNDS: int = int(os.getenv("GX_MAX_ROUNDS", "200"))
    DUELS_PER_ROUND: int = int(os.getenv("GX_DUELS_PER_ROUND", "3"))

    # BTL & CI
    CI_MIN_ROUNDS: int = int(os.getenv("GX_CI_MIN_ROUNDS", "6"))
    CI_BOOTSTRAP_SAMPLES: int = int(os.getenv("GX_CI_BOOTSTRAP", "200"))
    CI_SEPARATION_MIN: float = float(os.getenv("GX_CI_SEP_MIN", "0.05"))
    PAIR_SAMPLE_LIMIT: int = int(os.getenv("GX_PAIR_SAMPLE_LIMIT", "120"))
    BTL_FLOOR: float = float(os.getenv("GX_BTL_FLOOR", "1e-6"))
    BTL_ITERS: int = int(os.getenv("GX_BTL_ITERS", "3"))

    # UCB
    UCB_WEIGHT: float = float(os.getenv("GX_UCB_WEIGHT", "0.5"))

    # Dawid–Skene (stub weights)
    DS_INIT_ACC: float = float(os.getenv("GX_DS_INIT_ACC", "0.6"))
    DS_MIN_WEIGHT: float = float(os.getenv("GX_DS_MIN_WEIGHT", "0.1"))
    DS_EMA_ALPHA: float = float(os.getenv("GX_DS_EMA_ALPHA", "0.1"))

    # Claims
    MIN_CLAIM_CHARS: int = int(os.getenv("GX_MIN_CLAIM_CHARS", "10"))

    # Confidence default
    CONFIDENCE_THRESHOLD: float = float(os.getenv("GX_CONFIDENCE_THR", "0.95"))

    # LLM referee
    USE_REAL_LLM: bool = os.getenv("GX_USE_REAL_LLM", "false").lower() == "true"
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL: str = os.getenv("GX_LLM_MODEL", "claude-3-5-sonnet-20240620")
    LLM_MAX_TOKENS: int = int(os.getenv("GX_LLM_MAX_TOKENS", "1024"))


CONFIG = Config()

# ------------------------------------------------------------------------------
# Schemas (pydantic)
# ------------------------------------------------------------------------------

class ClaimScore(BaseModel):
    cid: str
    score: float
    ci_low: float = 0.0
    ci_high: float = 1.0

class TimelineItem(BaseModel):
    run_id: str
    round_index: int
    convergence_score: float
    best_claim_cid: Optional[str] = None
    best_claim_text: Optional[str] = None
    summary: Optional[str] = None
    top_claims: List[ClaimScore] = []

class QueryIn(BaseModel):
    query: str = Field(..., min_length=3)
    budget: conint(ge=1, le=CONFIG.MAX_BUDGET) = CONFIG.DEFAULT_BUDGET
    confidence_threshold: confloat(ge=0.5, le=0.999) = CONFIG.CONFIDENCE_THRESHOLD

class QueryOut(BaseModel):
    run_id: str
    query: str
    best_claim: str
    confidence: float
    rounds: int
    total_duels: int
    stop_reason: str

# ------------------------------------------------------------------------------
# Transparency Ledger (structured logs)
# ------------------------------------------------------------------------------

logger = logging.getLogger("gatewayx")
logging.basicConfig(level=logging.INFO)

class TransparencyLedger:
    """Structured logging of all key events (claims, duels, rounds, final result)."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def log(self, event_type: str, payload: Dict[str, Any]):
        evt = {
            "ts": round(time.time(), 3),
            "type": event_type,
            **payload,
        }
        self.events.append(evt)
        logger.info("ledger_event=%s", json.dumps(evt, separators=(",", ":")))

LEDGER = TransparencyLedger()

# ------------------------------------------------------------------------------
# PromptVault (strict JSON prompts)
# ------------------------------------------------------------------------------

class PromptVault:
    @staticmethod
    def duel_prompt(query: str, a: str, b: str) -> str:
        return (
            "You are a strict referee. Compare two answers for accuracy, coherence, and relevance.\n"
            'Return STRICT JSON only: {"winner":"A"|"B","factuality":0..1,"coherence":0..1,"note":"<=20 tokens"}\n'
            f"Query: {query}\nA: {a}\nB: {b}\n"
        )

    @staticmethod
    def validate_response(response: str) -> Dict[str, Any]:
        try:
            obj = json.loads(response)
            if obj.get("winner") not in ("A", "B"):
                raise ValueError("invalid winner")
            return {
                "winner": obj["winner"],
                "factuality": float(obj.get("factuality", 0.5)),
                "coherence": float(obj.get("coherence", 0.5)),
                "note": str(obj.get("note", ""))[:60],
            }
        except Exception:
            return {"winner": random.choice(["A", "B"]), "factuality": 0.5, "coherence": 0.5, "note": "fallback"}

# ------------------------------------------------------------------------------
# DuelScheduler (referee; async Anthropic or mock)
# ------------------------------------------------------------------------------

JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)

class DuelScheduler:
    def __init__(self, config: Config):
        self.config = config
        self.memo: Dict[str, Dict[str, Any]] = {}
        self.client = None
        if self.config.USE_REAL_LLM and AsyncAnthropic and self.config.ANTHROPIC_API_KEY:
            self.client = AsyncAnthropic(api_key=self.config.ANTHROPIC_API_KEY)

    async def initial_round(self, run_id: str, query: str, engines: List[str], num_claims: int = 5) -> List[str]:
        """
        For a prototype, synthesize 'n' diverse answers. In a multi-engine setup,
        you could actually call each engine here and merge responses.
        """
        claims = [f"[{eng}] Answer emphasizing aspect {i} for: {query}" for i, eng in zip(range(1, num_claims+1), engines * 10)]
        LEDGER.log("initial_claims", {"run_id": run_id, "claims": claims})
        return claims

    async def schedule_duel(self, run_id: str, query: str, a: str, b: str, playbook: str) -> Dict[str, Any]:
        key = self._cache_key(query, a, b, playbook)
        if key in self.memo:
            duel = self.memo[key]
            LEDGER.log("duel_cached", {"run_id": run_id, **duel})
            return duel

        prompt = PromptVault.duel_prompt(query, a, b)
        text = await self._call_llm(prompt) if self.client else self._mock_referee(a, b)
        # robust JSON extract
        m = JSON_OBJ_RE.search(text or "")
        parsed = PromptVault.validate_response(m.group(0) if m else "{}")

        duel = {"a": a, "b": b, "result": parsed, "playbook": playbook}
        self.memo[key] = duel
        LEDGER.log("duel_result", {"run_id": run_id, **duel})
        return duel

    async def _call_llm(self, prompt: str) -> str:
        msg = await self.client.messages.create(
            model=self.config.LLM_MODEL,
            max_tokens=self.config.LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text if msg and msg.content else "{}"

    def _mock_referee(self, a: str, b: str) -> str:
        # Simple, slightly noisy rule: longer answer wins, 10% flip chance
        win = "A" if len(a) >= len(b) else "B"
        if random.random() < 0.1:
            win = "A" if win == "B" else "B"
        return json.dumps({"winner": win, "factuality": 0.8, "coherence": 0.85, "note": "mock"})

    def _cache_key(self, query: str, a: str, b: str, playbook: str) -> str:
        return hashlib.sha256("|".join([query, a, b, playbook]).encode("utf-8")).hexdigest()

# ------------------------------------------------------------------------------
# BTL Ranker (MM updates + bootstrap CIs)
# ------------------------------------------------------------------------------

class BTLRanker:
    def __init__(self, config: Config):
        self.config = config
        self.theta: Dict[str, float] = {}
        self.wins: Dict[Tuple[str, str], int] = {}
        self.duels: List[Tuple[str, str, str]] = []
        self._rounds = 0
        self.cis: Dict[str, Tuple[float, float]] = {}

    def add_claims(self, claims: List[str]):
        for c in claims:
            self.theta.setdefault(c, 1.0 / max(1, len(claims)))
        self._normalize()

    def update(self, duel: Dict[str, Any]):
        a, b, w = duel["a"], duel["b"], duel["result"]["winner"]
        if w not in ("A", "B"):
            return
        key = (a, b) if w == "A" else (b, a)
        self.wins[key] = self.wins.get(key, 0) + 1
        self.duels.append((a, b, w))
        self._mm_update()
        self._rounds += 1

    def _mm_update(self):
        for _ in range(self.config.BTL_ITERS):
            denom = {i: 0.0 for i in self.theta}
            wins = {i: 0.0 for i in self.theta}
            for (i, j), w_ij in self.wins.items():
                n_ij = w_ij + self.wins.get((j, i), 0)
                s = self.theta[i] + self.theta[j]
                if s <= 0:
                    continue
                denom[i] += n_ij / s
                denom[j] += n_ij / s
                wins[i] += w_ij
                wins[j] += self.wins.get((j, i), 0)
            for i in self.theta:
                if denom[i] > 0:
                    self.theta[i] = max(self.config.BTL_FLOOR, self.theta[i] * (wins[i] / denom[i]))
        self._normalize()

    def _normalize(self):
        z = sum(self.theta.values()) or 1.0
        for k in self.theta:
            self.theta[k] /= z

    def select_k_informative_pairs(self, claims: List[str], k: int) -> List[Tuple[str, str]]:
        # sample pairs, score by entropy, pick k non-overlapping
        pairs = list(combinations(claims, 2))
        random.shuffle(pairs)
        sample = pairs[: min(len(pairs), self.config.PAIR_SAMPLE_LIMIT)]
        scored: List[Tuple[Tuple[str, str], float]] = []
        for a, b in sample:
            p = self._p_i_gt_j(a, b)
            h = -(p * math.log(p + 1e-12) + (1 - p) * math.log(1 - p + 1e-12))
            scored.append(((a, b), h))
        scored.sort(key=lambda x: x[1], reverse=True)
        chosen, used = [], set()
        for (a, b), _ in scored:
            if len(chosen) >= k:
                break
            if a in used or b in used:
                continue
            chosen.append((a, b))
            used.add(a); used.add(b)
        return chosen or ([(claims[0], claims[-1])] if claims else [])

    def compute_cis(self, n_bootstrap: int):
        if len(self.duels) < 2:
            self.cis = {c: (0.0, 1.0) for c in self.theta}
            return
        samples: List[Dict[str, float]] = []
        for _ in range(n_bootstrap):
            tmp = BTLRanker(self.config)
            tmp.add_claims(list(self.theta.keys()))
            for _ in range(len(self.duels)):
                a, b, w = random.choice(self.duels)
                tmp.update({"a": a, "b": b, "result": {"winner": w}})
            samples.append(tmp.theta.copy())
        self.cis = {}
        for c in self.theta:
            vals = sorted(s[c] for s in samples)
            lo = vals[int(0.025 * (len(vals) - 1))]
            hi = vals[int(0.975 * (len(vals) - 1))]
            self.cis[c] = (lo, hi)

    def ci_gap(self) -> float:
        if len(self.theta) < 2 or not self.cis:
            return 0.0
        ranked = sorted(self.theta.items(), key=lambda kv: kv[1], reverse=True)
        top, second = ranked[0][0], ranked[1][0]
        top_lo, _ = self.cis.get(top, (0.0, 1.0))
        _, sec_hi = self.cis.get(second, (0.0, 1.0))
        return max(0.0, top_lo - sec_hi)

    def get_confidence_proxy(self) -> float:
        vals = sorted(self.theta.values(), reverse=True)
        if len(vals) < 2:
            return 1.0
        margin = (vals[0] - vals[1]) / (vals[0] + vals[1] + 1e-12)
        rfac = min(1.0, self._rounds / max(5.0, float(CONFIG.CI_MIN_ROUNDS)))
        return max(0.0, min(1.0, margin * rfac))

    def best_claim(self) -> str:
        return max(self.theta, key=self.theta.get) if self.theta else ""

    def snapshot_state(self) -> Dict[str, float]:
        return dict(self.theta)

    def rounds(self) -> int:
        return self._rounds

    def _p_i_gt_j(self, i: str, j: str) -> float:
        t = self.theta[i] + self.theta[j]
        return self.theta[i] / t if t > 0 else 0.5

# ------------------------------------------------------------------------------
# Playbook Selector (UCB)
# ------------------------------------------------------------------------------

class PlaybookSelector:
    def __init__(self, config: Config):
        self.config = config
        self.arms = ["SelfConsistency", "Debate", "EvidenceFirst", "FocusOnDisputes"]
        self.n = {a: 0 for a in self.arms}
        self.mu = {a: 0.0 for a in self.arms}
        self.t = 0

    def choose_playbook(self, state: Dict[str, float], round_num: int, total_budget: int) -> str:
        self.t += 1
        phase = round_num / max(1, total_budget)
        pri = {
            "SelfConsistency": 0.25 if phase < 0.5 else 0.30,
            "Debate": 0.40 if phase < 0.5 else 0.25,
            "EvidenceFirst": 0.20,
            "FocusOnDisputes": 0.15 if phase < 0.7 else 0.25,
        }
        scores = {}
        for a in self.arms:
            # ensure each arm is explored at least once
            if self.n[a] == 0:
                return a
            bonus = math.sqrt(math.log(self.t + 1.0) / (self.n[a] + 1.0))
            scores[a] = pri[a] + self.config.UCB_WEIGHT * (self.mu[a] + bonus)
        return max(scores, key=scores.get)

    def update_performance(self, arm: str, reward: float):
        self.n[arm] += 1
        self.mu[arm] += (reward - self.mu[arm]) / self.n[arm]

# ------------------------------------------------------------------------------
# StopConditionEvaluator (CI-separation + confidence + max rounds)
# ------------------------------------------------------------------------------

class StopConditionEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self.last_reason = "running"

    def should_stop(self, btl: BTLRanker, target_confidence: float) -> bool:
        if btl.rounds() >= self.config.MAX_ROUNDS:
            self.last_reason = "max_rounds"; return True
        if btl.rounds() >= self.config.CI_MIN_ROUNDS and btl.ci_gap() >= self.config.CI_SEPARATION_MIN:
            self.last_reason = "ci_separation"; return True
        if btl.get_confidence_proxy() >= target_confidence:
            self.last_reason = "confidence_threshold"; return True
        self.last_reason = "running"
        return False

# ------------------------------------------------------------------------------
# HTML Report (inline SVG sparkline)
# ------------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    try:
        return f"{100.0 * x:.1f}%"
    except Exception:
        return "–"

def _build_convergence_svg(scores: List[float], width: int = 560, height: int = 120, pad: int = 10) -> str:
    if not scores:
        return '<svg width="0" height="0" role="img" aria-label="No data"></svg>'
    def y(v: float) -> int:
        v = max(0.0, min(1.0, float(v)))
        return pad + int((1.0 - v) * (height - 2 * pad))
    def x(i: int) -> int:
        n = len(scores)
        return pad + int(i * (width - 2 * pad) / max(1, n - 1))
    pts = " ".join(f"{x(i)},{y(v)}" for i, v in enumerate(scores))
    baseline_y = y(0.5)
    target_y = y(0.9)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Convergence">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>'
        f'<line x1="{pad}" y1="{baseline_y}" x2="{width-pad}" y2="{baseline_y}" stroke="#ddd" stroke-dasharray="4,4"/>'
        f'<line x1="{pad}" y1="{target_y}" x2="{width-pad}" y2="{target_y}" stroke="#e0e0ff" stroke-dasharray="3,3"/>'
        f'<polyline points="{pts}" fill="none" stroke="#2f6fdd" stroke-width="2" vector-effect="non-scaling-stroke" />'
        '</svg>'
    )

def render_timeline_html(run_id: str, items: List[TimelineItem]) -> str:
    rounds = len(items)
    final_score = items[-1].convergence_score if items else 0.0
    svg = _build_convergence_svg([it.convergence_score for it in items])
    rows = []
    for it in items:
        chips = ""
        if it.top_claims:
            chips = "<div class='chips'>" + " ".join(
                f"<span class='chip'>{html.escape(cs.cid)} · TS {_fmt_pct(cs.score)} · CI [{_fmt_pct(cs.ci_low)}–{_fmt_pct(cs.ci_high)}]</span>"
                for cs in it.top_claims[:5]
            ) + "</div>"
        rows.append(
            "<tr>"
            f"<td>{it.round_index}</td>"
            f"<td>{_fmt_pct(it.convergence_score)}</td>"
            f"<td>{html.escape(it.best_claim_cid or '—')}</td>"
            f"<td>{html.escape((it.best_claim_text or '—')[:240])}</td>"
            f"<td>{html.escape((it.summary or '—')[:320])}{chips}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Gateway X — Run {html.escape(run_id)}</title>
<style>
:root {{ --bg:#fff; --fg:#222; --muted:#666; --line:#e9e9e9; --chip:#f5f7fb; --chipfg:#2f3a4a; }}
body {{ margin:0; padding:24px; font:14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, Arial; color:var(--fg); background:var(--bg); }}
header {{ display:flex; align-items:baseline; gap:12px; margin-bottom:12px; }}
h1 {{ font-size:18px; margin:0; }}
.meta {{ color:var(--muted); font-size:13px; }}
.panel {{ border:1px solid var(--line); border-radius:10px; padding:16px; margin:16px 0; }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ padding:10px; border-top:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ background:#fafafa; border-top:none; font-weight:600; }}
.chips {{ margin-top:6px; display:flex; flex-wrap:wrap; gap:6px; }}
.chip {{ padding:4px 8px; border-radius:999px; background:var(--chip); color:var(--chipfg); border:1px solid #e6ecf5; font-size:12px; }}
code {{ background:#fafafa; border:1px solid var(--line); padding:2px 6px; border-radius:6px; }}
footer {{ margin-top: 24px; color: var(--muted); font-size: 12px; }}
</style>
</head><body>
<header>
  <h1>Gateway X — Run <code>{html.escape(run_id)}</code></h1>
  <div class="meta">Rounds: <strong>{rounds}</strong> · Final Convergence: <strong>{_fmt_pct(final_score)}</strong></div>
</header>
<section class="panel">
  <div class="meta">Convergence over rounds (grey=50%, blue dotted=90% target)</div>
  {svg}
</section>
<section class="panel">
  <div class="meta">Per-round truth response snapshot</div>
  <table>
    <thead><tr><th style="width:60px;">Round</th><th style="width:120px;">Convergence</th>
      <th style="width:160px;">Best Claim ID</th><th>Best Claim (excerpt)</th><th>Summary & Top Claims</th></tr></thead>
    <tbody>{''.join(rows) if rows else '<tr><td colspan="5">No rounds recorded.</td></tr>'}</tbody>
  </table>
</section>
<footer>Generated by Gateway X Timeline Reporter.</footer>
</body></html>"""

# ------------------------------------------------------------------------------
# Orchestrator (keeps timeline & aggregates round "truth response")
# ------------------------------------------------------------------------------

class Orchestrator:
    """
    Runs rounds until governance thresholds are met.
    For each round:
      1) Select informative pairs; run parallel duels.
      2) Update BTL; compute convergence score (truth score).
      3) Build a "truth response" (summary) + record top claims.
    Stores a per-run timeline for /timeline and /timeline/report.
    """

    def __init__(self, config: Config):
        self.config = config
        self.scheduler = DuelScheduler(config)
        self.btl = BTLRanker(config)
        self.selector = PlaybookSelector(config)
        self.stopper = StopConditionEvaluator(config)
        self.runs: Dict[str, List[TimelineItem]] = {}  # run_id -> timeline
        self.current_engines: List[str] = ["engine.alpha", "engine.beta", "engine.gamma", "engine.delta"]

    async def run(self, query: str, budget: int, confidence_threshold: float) -> Dict[str, Any]:
        run_id = hashlib.sha256(f"{time.time()}|{query}".encode()).hexdigest()[:12]
        timeline: List[TimelineItem] = []
        self.runs[run_id] = timeline

        # Round 0: initial claims
        claims = await self.scheduler.initial_round(run_id, query, engines=self.current_engines, num_claims=5)
        claims = self._filter_invalid_claims(claims)
        self.btl.add_claims(claims)

        total_duels = 0
        prev_conf = 0.0
        stop_reason = "budget_exhausted"

        for round_idx in range(1, budget + 1):
            playbook = self.selector.choose_playbook(self.btl.snapshot_state(), round_idx, budget)
            pairs = self.btl.select_k_informative_pairs(claims, k=self.config.DUELS_PER_ROUND)

            # Run duels (parallel)
            tasks = [self.scheduler.schedule_duel(run_id, query, a, b, playbook) for a, b in pairs]
            results = []
            if tasks:
                # naive async gather to avoid blocking
                import asyncio
                results = await asyncio.gather(*tasks)

            for duel in results:
                self.btl.update(duel)
            total_duels += len(results)

            # compute CIs if enough rounds elapsed
            if self.btl.rounds() >= self.config.CI_MIN_ROUNDS:
                self.btl.compute_cis(self.config.CI_BOOTSTRAP_SAMPLES)

            # reward for UCB = Δ confidence proxy
            cur_conf = self.btl.get_confidence_proxy()
            self.selector.update_performance(playbook, max(0.0, cur_conf - prev_conf))
            prev_conf = cur_conf

            # assemble truth response (summary) for this round
            best = self.btl.best_claim()
            # top-5 claims w/ CI
            ranked = sorted(self.btl.snapshot_state().items(), key=lambda kv: kv[1], reverse=True)
            top = [
                ClaimScore(
                    cid=cid,
                    score=score,
                    ci_low=self.btl.cis.get(cid, (0.0, 1.0))[0],
                    ci_high=self.btl.cis.get(cid, (0.0, 1.0))[1],
                )
                for cid, score in ranked[:5]
            ]
            truth_summary = _build_truth_summary(best, top)

            item = TimelineItem(
                run_id=run_id,
                round_index=round_idx,
                convergence_score=cur_conf,
                best_claim_cid=best,
                best_claim_text=best,
                summary=truth_summary,
                top_claims=top,
            )
            timeline.append(item)
            LEDGER.log("round_summary", item.model_dump())

            if self.stopper.should_stop(self.btl, confidence_threshold):
                stop_reason = self.stopper.last_reason
                break

        out = {
            "run_id": run_id,
            "query": query,
            "best_claim": self.btl.best_claim(),
            "confidence": self.btl.get_confidence_proxy(),
            "rounds": len(timeline),
            "total_duels": total_duels,
            "stop_reason": stop_reason,
        }
        LEDGER.log("final_result", out)
        return out

    def get_timeline(self, run_id: str) -> List[TimelineItem]:
        return self.runs.get(run_id, [])

    def _filter_invalid_claims(self, claims: List[str]) -> List[str]:
        seen = set()
        clean = []
        for c in claims:
            s = "".join(ch for ch in c.lower().strip() if ch.isalnum() or ch.isspace())
            if len(s) < self.config.MIN_CLAIM_CHARS or s in seen:
                continue
            seen.add(s)
            clean.append(c.strip())
        return clean or claims

def _build_truth_summary(best_cid: Optional[str], top: List[ClaimScore]) -> str:
    """
    Tiny, deterministic summary that:
      - names current best claim,
      - cites up to 3 runner-up IDs (with scores),
      - notes divergence (CI overlaps) when present.
    """
    if not best_cid:
        return "No consensus yet."
    others = [cs for cs in top if cs.cid != best_cid]
    diverge = [cs for cs in others if cs.ci_high >= next((t.ci_low for t in top if t.cid == best_cid), 0.0)]
    parts = [f"Truth response selects {best_cid} as current best."]
    if others:
        parts.append("Next contenders: " + ", ".join(f"{cs.cid} (TS {cs.score:.2f})" for cs in others[:3]) + ".")
    if diverge:
        parts.append("Some divergence remains (CI overlap with best).")
    return " ".join(parts)

# ------------------------------------------------------------------------------
# FastAPI app & routes
# ------------------------------------------------------------------------------

app = FastAPI(title="Gateway X (single-file)", version="30.2")
ORCH = Orchestrator(CONFIG)

@app.post("/query", response_model=QueryOut)
async def handle_query(body: QueryIn):
    try:
        res = await ORCH.run(body.query, body.budget, body.confidence_threshold)
        return QueryOut(**res)
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/timeline/{run_id}")
async def get_timeline(run_id: str):
    try:
        items = ORCH.get_timeline(run_id)
        return [it.model_dump() for it in items]
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/timeline/{run_id}/report", response_class=HTMLResponse)
async def get_timeline_report(run_id: str):
    try:
        items = ORCH.get_timeline(run_id)
        html_doc = render_timeline_html(run_id, [TimelineItem(**it.model_dump()) for it in items])
        return HTMLResponse(content=html_doc, status_code=200)
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

# Root
@app.get("/")
async def root():
    return {"service": "Gateway X", "version": "30.2", "use_real_llm": CONFIG.USE_REAL_LLM}
