"""检索内部校验原语，不改变公开接缝。"""
from urllib.parse import urlsplit, urlunsplit


def canonical_url(url: object) -> str:
    """只接受无凭据 HTTP(S) URL；去 fragment/默认端口，保留路径和查询语义。"""
    if not isinstance(url, str) or len(url) > 8192 or any(c.isspace() for c in url):
        return ""
    try:
        p = urlsplit(url)
        if p.scheme not in ("http", "https") or not p.hostname or p.username or p.password:
            return ""
        host = p.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        port = p.port
        if port and (p.scheme, port) not in (("http", 80), ("https", 443)):
            host += f":{port}"
        return urlunsplit((p.scheme, host, p.path or "/", p.query, ""))
    except ValueError:
        return ""
