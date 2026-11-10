"""搜索后端可安全公开的错误元数据；不携带服务响应正文或凭据。"""
from __future__ import annotations


class SearchBackendError(RuntimeError):
    """保持RuntimeError兼容性，使用固定分类和可选HTTP状态码。"""

    REASONS = frozenset({"http_error", "network_timeout", "connection_error",
                         "invalid_json", "invalid_response", "response_too_large",
                         "extract_failed", "unexpected_error"})

    def __init__(self, reason: str, *, http_status: int | None = None) -> None:
        self.reason = reason if reason in self.REASONS else "unexpected_error"
        self.http_status = (http_status if type(http_status) is int
                            and 100 <= http_status <= 599 else None)
        status = f"; http_status={self.http_status}" if self.http_status is not None else ""
        super().__init__(f"search backend {self.reason}{status}; details redacted")
