from urllib.parse import urlparse


TRUSTED_DOMAINS = {
    "bbc.com",
    "reuters.com",
    "ndtv.com",
    "thehindu.com",
}


def normalize_domain(url: str) -> str:
    raw = (url or "").strip().lower()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    domain = parsed.netloc.replace("www.", "")
    return domain


def is_trusted_source(url: str) -> bool:
    domain = normalize_domain(url)
    if not domain:
        return False
    return any(domain == trusted or domain.endswith("." + trusted) for trusted in TRUSTED_DOMAINS)
