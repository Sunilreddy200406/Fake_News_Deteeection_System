from __future__ import annotations

from typing import Dict, Optional

from src.decision_engine import make_final_decision
from src.portal_verifier import fetch_official_articles
from src.preprocess import extract_entities, extract_keywords, preprocess_text
from src.similarity import embedding_similarity_score, tfidf_similarity_score
from src.source_verifier import is_trusted_source, normalize_domain


def _result_category(final_label: str) -> str:
    text = final_label.lower()
    if "unverified" in text or "suspicious" in text:
        return "Unverified"
    if "fake" in text:
        return "Fake"
    if "verified official source" in text or text.startswith("real"):
        return "Real"
    return "Unverified"


def _method_label(decision_path: str) -> str:
    if decision_path in {"trusted_source_url", "official_portal_verification"}:
        return "Official Source Comparison"
    return "Machine Learning"


def analyze_news(text: str, source_url: str, model_bundle: Dict[str, object]) -> Dict[str, object]:
    source_domain = normalize_domain(source_url) if source_url else None
    cleaned_for_model = preprocess_text(text)
    keywords = extract_keywords(text)
    entities = extract_entities(text)

    trusted_source = is_trusted_source(source_url) if source_url else False
    articles = fetch_official_articles(text)
    article_texts = [a.combined_text for a in articles]

    tfidf_score, tfidf_idx = tfidf_similarity_score(text, article_texts)
    emb_score, emb_idx = embedding_similarity_score(text, article_texts)
    similarity_score = max(tfidf_score, emb_score)
    best_idx = tfidf_idx if tfidf_score >= emb_score else emb_idx

    matched_article: Optional[Dict[str, str]] = None
    if best_idx >= 0 and best_idx < len(articles):
        matched = articles[best_idx]
        matched_article = {
            "title": matched.title,
            "link": matched.link,
            "source": matched.source_domain,
            "similarity_score": round(similarity_score, 4),
        }

    # If URL itself is trusted, we still keep similarity details for explanation.
    if trusted_source:
        final_label = "Real News (Verified Official Source)"
        decision_path = "trusted_source_url"
        return {
            "result": _result_category(final_label),
            "verification_method": _method_label(decision_path),
            "final_label": "Real News (Verified Official Source)",
            "prediction": "Real",
            "reasoning": "Input source URL belongs to trusted official domain.",
            "decision_path": decision_path,
            "confidence": 1.0,
            "source_domain": source_domain,
            "keywords": keywords,
            "entities": entities,
            "similarity": {
                "tfidf": round(tfidf_score, 4),
                "embedding": round(emb_score, 4),
                "best": round(similarity_score, 4),
            },
            "matched_article": matched_article,
            "official_articles_checked": len(articles),
        }

    vectorized = model_bundle["vectorizer"].transform([cleaned_for_model])
    pred_num = int(model_bundle["model"].predict(vectorized)[0])
    pred_label = model_bundle.get("label_map", {0: "Fake", 1: "Real"}).get(pred_num, str(pred_num))
    ml_confidence = 0.5
    if hasattr(model_bundle["model"], "predict_proba"):
        ml_probs = model_bundle["model"].predict_proba(vectorized)[0]
        ml_confidence = float(max(ml_probs))

    decision = make_final_decision(
        portal_score=similarity_score,
        portal_threshold=0.62,
        ml_label=pred_label,
        ml_confidence=ml_confidence,
        matched_article=matched_article,
    )

    return {
        "result": _result_category(decision["final_label"]),
        "verification_method": _method_label(decision["decision_path"]),
        "final_label": decision["final_label"],
        "prediction": pred_label,
        "reasoning": decision["reasoning"],
        "decision_path": decision["decision_path"],
        "confidence": ml_confidence,
        "source_domain": source_domain,
        "keywords": keywords,
        "entities": entities,
        "similarity": {
            "tfidf": round(tfidf_score, 4),
            "embedding": round(emb_score, 4),
            "best": round(similarity_score, 4),
        },
        "matched_article": decision["matched_article"],
        "official_articles_checked": len(articles),
    }
