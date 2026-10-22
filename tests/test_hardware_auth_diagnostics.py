"""Exercise HTTP authentication diagnostics without exposing authentication data."""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from aiohttp import ClientSession, web
from multidict import CIMultiDict
from yarl import URL

from custom_components.loxone.hardware import api as api_module
from custom_components.loxone.hardware.api import LoxoneHardwareApi, LoxoneHardwareError


def make_api(session):
    """Create an isolated client with deliberately recognizable test credentials."""
    return LoxoneHardwareApi(
        session,
        host="127.0.0.1",
        port=80,
        username="test-user",
        password="private-password",
        verify_ssl=True,
        structure={},
    )


def response(url, authorization=None, status=401, history=()):
    """Model the actual request headers rather than the configured defaults."""
    headers = CIMultiDict()
    if authorization is not None:
        headers["Authorization"] = authorization
    return SimpleNamespace(
        status=status,
        history=history,
        request_info=SimpleNamespace(real_url=URL(url), headers=headers, method="GET"),
    )


def diagnostic(caplog):
    """Extract the structured, safe warning payload."""
    record = next(item for item in caplog.records if "authentication diagnostic:" in item.message)
    return json.loads(record.message.split("diagnostic: ", 1)[1])


@pytest.mark.parametrize("kind", ["missing", "different", "matching"])
def test_actual_authorization_flags_and_redaction(kind, caplog):
    """Only presence and equality booleans survive arbitrary request secrets."""
    client = make_api(Mock())
    expected = client.headers["Authorization"]
    header = {"missing": None, "different": "Bearer private-token", "matching": expected}[kind]
    reply = response("http://user:private-password@127.0.0.1/data/status?token=private-query", header)
    with caplog.at_level(logging.WARNING), pytest.raises(LoxoneHardwareError) as caught:
        client._check_authentication("data/status?token=private-query", reply)
    hop = diagnostic(caplog)["hops"][0]
    assert hop == {
        "endpoint": "data/status",
        "status": 401,
        "method": "GET",
        "same_origin": True,
        "auth_present": kind != "missing",
        "auth_matches_expected": kind == "matching",
    }
    output = caplog.text + str(caught.value)
    for secret in ("private-password", "private-query", "private-token", expected, "test-user"):
        assert secret not in output


def test_redirect_chain_hides_unknown_paths_and_foreign_hosts(caplog):
    """Cross-origin header removal remains visible without disclosing destinations."""
    client = make_api(Mock())
    first = response("http://127.0.0.1/data/status", client.headers["Authorization"], 302)
    last = response("https://private-host/private-token?secret=private-query", history=(first,))
    with caplog.at_level(logging.WARNING), pytest.raises(LoxoneHardwareError):
        client._check_authentication("data/status", last)
    hops = diagnostic(caplog)["hops"]
    assert [hop["status"] for hop in hops] == [302, 401]
    assert [hop["same_origin"] for hop in hops] == [True, False]
    assert [hop["auth_matches_expected"] for hop in hops] == [True, False]
    assert hops[-1]["endpoint"] == "<other>"
    assert all(secret not in caplog.text for secret in ("private-host", "private-token", "private-query"))


def test_warning_rate_limit_does_not_hide_changed_evidence(caplog, monkeypatch):
    """Repeated failures are bounded, but changed headers are captured immediately."""
    now = 100.0
    monkeypatch.setattr(api_module, "monotonic", lambda: now)
    client = make_api(Mock())
    reply = response("http://127.0.0.1/data/status")
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            with pytest.raises(LoxoneHardwareError):
                client._check_authentication("data/status", reply)
        assert len(caplog.records) == 1
        reply.request_info.headers["Authorization"] = client.headers["Authorization"]
        with pytest.raises(LoxoneHardwareError):
            client._check_authentication("data/status", reply)
        assert len(caplog.records) == 2
        now += 60
        with pytest.raises(LoxoneHardwareError):
            client._check_authentication("data/status", reply)
        assert len(caplog.records) == 3


def test_channel_endpoint_and_bounded_diagnostic_cache(caplog):
    """Retain physical channel addresses while bounding persistent diagnostic data."""
    client = make_api(Mock())
    with caplog.at_level(logging.WARNING):
        for index in range(api_module.MAX_AUTH_DIAGNOSTICS + 1):
            path = f"jdev/sps/io/0C000001.{index:06X}.AI2"
            with pytest.raises(LoxoneHardwareError):
                client._check_authentication(path, response(f"http://127.0.0.1/{path}"))
    assert len(client._auth_diagnostic_times) == api_module.MAX_AUTH_DIAGNOSTICS
    assert diagnostic(caplog)["endpoint"] == "jdev/sps/io/0C000001.000000.AI2"


@pytest.mark.parametrize("method", ["_request_json", "_request_text"])
@pytest.mark.parametrize("redirect", [False, True])
def test_real_http_401_and_success(method, redirect, caplog):
    """Run both request paths against a local server, including a real redirect."""

    async def exercise():
        app = web.Application()

        async def reject(request):
            return web.Response(status=401, reason="private-reason", text="private-body")

        async def route(request):
            if redirect:
                raise web.HTTPFound("/private-token?token=private-query")
            return await reject(request)

        async def ok(request):
            return web.json_response({"ok": True})

        app.router.add_get("/data/status", route)
        app.router.add_get("/private-token", reject)
        app.router.add_get("/ok", ok)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            async with ClientSession() as session:
                client = make_api(session)
                client.base_url = f"http://127.0.0.1:{port}"
                with pytest.raises(LoxoneHardwareError) as caught:
                    await getattr(client, method)("data/status")
                assert caught.value.__suppress_context__
                assert caught.value.__cause__ is None
                assert caught.value.__context__ is None
                output = caplog.text + str(caught.value)
                for secret in (
                    "private-password",
                    "private-reason",
                    "private-body",
                    "private-query",
                    "private-token",
                    client.headers["Authorization"],
                ):
                    assert secret not in output
                hops = diagnostic(caplog)["hops"]
                assert len(hops) == (2 if redirect else 1)
                assert all(hop["auth_matches_expected"] for hop in hops)
                result = await getattr(client, method)("ok")
                assert (result if method == "_request_json" else json.loads(result)) == {"ok": True}
        finally:
            await runner.cleanup()

    with caplog.at_level(logging.WARNING, logger=api_module.__name__):
        asyncio.run(exercise())


@pytest.mark.parametrize("status", [200, 403, 500])
def test_other_statuses_do_not_emit_authentication_diagnostic(status, caplog):
    """Do not change the treatment of successful or non-401 responses."""
    client = make_api(Mock())
    client._check_authentication("data/status", response("http://127.0.0.1/data/status", status=status))
    assert not caplog.records
