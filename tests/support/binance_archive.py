"""Test helpers that fake data.binance.vision over `httpx.MockTransport`. No network.

Typical use:

    zip_bytes = build_zip({"BTCUSDT-1d-2024-01.csv": csv})
    server = ArchiveServer(
        {
            file.url: (200, zip_bytes),
            file.checksum_url: (200, build_checksum(zip_bytes, "BTCUSDT-1d-2024-01.zip")),
        }
    )
    with server.client() as client:
        ...
    assert server.requested_urls == [file.checksum_url, file.url]

A response value is either one `(status, body)` pair (served every time) or a list of pairs
served in order, the last one repeating. A URL that is not in the dict answers 404.
`ArchiveServer.failing()` builds a server that fails the test on any request.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Mapping

import httpx

Response = tuple[int, bytes]


def build_zip(members: Mapping[str, bytes]) -> bytes:
    """Return a zip archive holding `members` (name -> content), in insertion order."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


def build_csv_zip(csv: bytes, name: str = "data.csv") -> bytes:
    """Return a zip archive holding exactly one CSV member."""
    return build_zip({name: csv})


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_checksum(zip_bytes: bytes, filename: str, *, trailing_newline: bool = True) -> bytes:
    """Return the text of a Binance `.CHECKSUM` file: "<sha256 hex>  <filename>"."""
    text = f"{sha256_hex(zip_bytes)}  {filename}"
    return (text + ("\n" if trailing_newline else "")).encode()


def _as_list(value: Response | list[Response]) -> list[Response]:
    return value if isinstance(value, list) else [value]


class ArchiveServer:
    """Serves a `{url: response}` dict and records every request it receives."""

    def __init__(
        self,
        responses: Mapping[str, Response | list[Response]],
        *,
        fail_on_request: bool = False,
    ) -> None:
        self._responses: dict[str, list[Response]] = {
            url: _as_list(value) for url, value in responses.items()
        }
        self._served: dict[str, int] = {}
        self._fail_on_request = fail_on_request
        self.requested_urls: list[str] = []

    @classmethod
    def failing(cls) -> ArchiveServer:
        """A server that raises AssertionError on any request (proves a cache hit)."""
        return cls({}, fail_on_request=True)

    @property
    def request_count(self) -> int:
        return len(self.requested_urls)

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if self._fail_on_request:
            raise AssertionError(f"unexpected request: {request.method} {url}")
        self.requested_urls.append(url)
        queue = self._responses.get(url)
        if not queue:
            return httpx.Response(404)
        index = self._served.get(url, 0)
        self._served[url] = index + 1
        status, body = queue[min(index, len(queue) - 1)]
        return httpx.Response(status, content=body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))
