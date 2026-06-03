"""LLM-assisted candidate stability scoring module.

Leverages pluggable LLM API providers (Gemini, OpenAI, Claude) to perform
structured biological evidence analysis, rating candidate stability and quality.
Includes a deterministic fallback model for offline execution and tests.
"""

from __future__ import annotations

import json
import os
import re
import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config
from .io_utils import get_logger

log = get_logger("wp3.llm_scoring")


SYSTEM_PROMPT = """You are a Principal AI Scientist and Enzyme Engineer at Cellarim Labs.
Your task is to evaluate structured biological and evolutionary evidence for a set of reconstructed ancestral enzyme candidates.
You will receive a structured JSON table containing candidates along with their biophysical and phylogenetic features.

For each candidate, evaluate its overall developability, fold probability, and thermostability potential based on:
1. Manifold Proximity (embedding_similarity): cosine proximity to high-value natural lipases.
2. Active Site integrity (motif_preservation and active_site_preservation).
3. Family consensus conservation (conservation_score).
4. Evolutionary distance (mismatch vs natural reference).
5. ASR Shannon Entropy (uncertainty_entropy). Lower is better.
6. Fold confidence (fold_confidence, when structural predictions are available. 0.0 means pre-fold).

Generate a stability rating for each candidate on a scale of 0.0 to 10.0 (where 10.0 is exceptionally stable, and 0.0 is completely unstable/unfolded), a confidence score (0.0 to 1.0), and 3-4 bullet points of crisp biological reasoning.

Your response MUST be a strictly formatted JSON array containing only objects with these keys:
- candidate_id (string)
- stability_score (float, 0.0 to 10.0)
- confidence (float, 0.0 to 1.0)
- reasoning (list of strings)

Do NOT write any conversational text, explanations, or markdown fences in your response. Output raw JSON only."""


# ── Pluggable REST Callers ───────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def _call_gemini_api(prompt: str, model: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\nStructured Feature Table:\n{prompt}"}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    res = resp.json()
    return res["candidates"][0]["content"]["parts"][0]["text"]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def _call_openai_api(prompt: str, model: str, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Structured Feature Table:\n{prompt}"}
        ]
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    res = resp.json()
    return res["choices"][0]["message"]["content"]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def _call_claude_api(prompt: str, model: str, api_key: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0.1,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": f"Structured Feature Table:\n{prompt}"}
        ]
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    res = resp.json()
    return res["content"][0]["text"]


# ── Deterministic Fallback ───────────────────────────────────────────────────

def _generate_fallback_scores(feature_table_json: str) -> str:
    """Deterministic offline fallback simulating the LLM scientist output."""
    records = json.loads(feature_table_json)
    results = []
    
    for r in records:
        cid = r["candidate_id"]
        # Basic heuristic score out of 10.0
        score = (
            3.5 * r["embedding_similarity"]
            + 1.5 * r["motif_preservation"]
            + 1.0 * r["conservation_score"]
            - 1.0 * r["uncertainty_entropy"]
        )
        
        # Injects fold confidence if post-fold is available
        if r.get("fold_confidence", 0.0) > 0.0:
            score += 3.0 * r["fold_confidence"]
            stability = float(np.clip(score, 0.0, 10.0))
        else:
            # Scale sequence-only base to 10 max
            stability = float(np.clip(score * (10.0 / 6.0), 0.0, 10.0))
            
        confidence = float(np.clip(0.70 + 0.20 * r["embedding_similarity"], 0.0, 1.0))
        
        reasoning = []
        if r["embedding_similarity"] >= 0.95:
            reasoning.append(f"Strong whole-sequence ESM-2 manifold similarity ({r['embedding_similarity']:.2f}) to active extant anchors.")
        else:
            reasoning.append(f"Moderate family manifold proximity ({r['embedding_similarity']:.2f}).")
            
        if r["motif_preservation"] >= 1.0:
            reasoning.append("GxSxG catalytic nucleophile elbow fully preserved in its natural geometry.")
        else:
            reasoning.append("WARNING: Possible mutation or substitution in the catalytic motif.")
            
        if r["uncertainty_entropy"] <= 0.05:
            reasoning.append(f"Very high reconstruction confidence at this ancestral node (entropy={r['uncertainty_entropy']:.2f}).")
        else:
            reasoning.append(f"Moderate site-wise uncertainty (entropy={r['uncertainty_entropy']:.2f}) at ancestral state positions.")
            
        if r.get("fold_confidence", 0.0) > 0.0:
            reasoning.append(f"Structural confidence verified via ESMFold: pLDDT={int(r['fold_confidence'] * 100)}% (Rg={r.get('packing_density', 0.0):.2f}).")

        results.append({
            "candidate_id": cid,
            "stability_score": round(stability, 2),
            "confidence": round(confidence, 2),
            "reasoning": reasoning
        })
        
    return json.dumps(results)


# ── Public Entry Point ────────────────────────────────────────────────────────

def score_candidates_with_llm(feature_table_json: str, cfg: Config) -> pd.DataFrame:
    """Dispatch feature table to the selected LLM provider.

    Returns a DataFrame indexed by candidate_id with columns:
        * llm_stability_score (float [0, 10])
        * llm_confidence (float [0, 1])
        * llm_reasoning (semicolon-separated list of strings)
    """
    provider = cfg.llm.provider.lower()
    api_key = os.environ.get(cfg.llm.api_key_env_var, "")
    
    # Graceful fallback check
    if provider == "fallback" or not api_key:
        if not api_key and provider != "fallback":
            log.warning("missing_api_key_fallback", 
                        reason=f"Environment variable '{cfg.llm.api_key_env_var}' not found. Falling back to deterministic scoring.")
        log.info("llm_scoring_running_fallback", provider=provider)
        response_text = _generate_fallback_scores(feature_table_json)
    else:
        log.info("llm_scoring_request", provider=provider, model=cfg.llm.model)
        try:
            if provider == "gemini":
                response_text = _call_gemini_api(feature_table_json, cfg.llm.model, api_key)
            elif provider == "openai":
                response_text = _call_openai_api(feature_table_json, cfg.llm.model, api_key)
            elif provider == "claude":
                response_text = _call_claude_api(feature_table_json, cfg.llm.model, api_key)
            else:
                raise ValueError(f"Unknown LLM provider: {provider}")
        except Exception as e:
            log.warning("llm_api_failed_fallback", error=str(e))
            response_text = _generate_fallback_scores(feature_table_json)

    # Parse LLM JSON Output
    try:
        # Clean potential markdown fences from the response if any
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            # strip off ```json and ```
            clean_text = re.sub(r"^```[a-zA-Z]*\n", "", clean_text)
            clean_text = re.sub(r"\n```$", "", clean_text)
            clean_text = clean_text.strip()
            
        data = json.loads(clean_text)
        
        # Handle cases where LLM returns single object with a key or nested structure
        if isinstance(data, dict):
            # check if there's a nested list
            for key in ["candidates", "results", "scores", "data"]:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            if isinstance(data, dict):  # still dict, wrap in list
                data = [data]
                
        rows = []
        for item in data:
            cid = item.get("candidate_id")
            if not cid:
                continue
            
            # Extract reasoning list
            reasoning = item.get("reasoning", [])
            if isinstance(reasoning, list):
                reasoning_str = "; ".join(str(x) for x in reasoning)
            else:
                reasoning_str = str(reasoning)
                
            rows.append({
                "candidate_id": cid,
                "llm_stability_score": float(item.get("stability_score", 0.0)),
                "llm_confidence": float(item.get("confidence", 0.0)),
                "llm_reasoning": reasoning_str
            })
            
        df = pd.DataFrame(rows).set_index("candidate_id")
        log.info("llm_scoring_complete", candidates=len(df))
        return df
        
    except Exception as e:
        log.error("json_parse_failed", error=str(e), raw_output=response_text[:300])
        # Direct backup recovery: generate fallback scores
        backup_json = _generate_fallback_scores(feature_table_json)
        backup_data = json.loads(backup_json)
        rows = []
        for item in backup_data:
            rows.append({
                "candidate_id": item["candidate_id"],
                "llm_stability_score": float(item["stability_score"]),
                "llm_confidence": float(item["confidence"]),
                "llm_reasoning": "; ".join(item["reasoning"])
            })
        return pd.DataFrame(rows).set_index("candidate_id")
