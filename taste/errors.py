"""Stable domain failures shared by HTTP, MCP, and local policy."""


class DomainError(Exception):
    """A public-safe message supplied by policy, never a raw dependency error."""

    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

    def payload(self) -> dict[str, dict[str, str]]:
        return {"error": {"code": self.code, "message": self.message}}
