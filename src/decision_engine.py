from __future__ import annotations

from typing import Dict, Optional


def make_final_decision(
    portal_score: float,
    portal_threshold: float,
    ml_label: str,
    ml_confidence: float,
    matched_article: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    if portal_score >= portal_threshold:
        return {
            "final_label": "Real News (Verified Official Source)",
            "reasoning": "High similarity with official portal article.",
            "decision_path": "official_portal_verification",
            "matched_article": matched_article,
        }

    if ml_label.lower() == "fake" and ml_confidence >= 0.7:
        return {
            "final_label": "Fake News",
            "reasoning": "Official verification failed and ML confidence for fake is high.",
            "decision_path": "ml_secondary_check",
            "matched_article": matched_article,
        }

    if ml_label.lower() == "real" and ml_confidence >= 0.65:
        return {
            "final_label": "Real (Unverified)",
            "reasoning": "Official verification not found, but ML confidence indicates likely real.",
            "decision_path": "ml_secondary_check",
            "matched_article": matched_article,
        }

    return {
        "final_label": "Suspicious / Unverified News",
        "reasoning": "Neither official verification nor ML confidence was strong enough.",
        "decision_path": "ml_secondary_check",
        "matched_article": matched_article,
    }

