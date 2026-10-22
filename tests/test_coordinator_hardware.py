"""Regression tests for hardware polling configuration at setup."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.loxone import coordinator as coordinator_module
from custom_components.loxone.const import CONF_HARDWARE_BATTERY_INTERVAL


@pytest.mark.parametrize("configured_seconds", [None, 60, 900, 1800])
def test_setup_preserves_battery_interval_seconds(
    monkeypatch: pytest.MonkeyPatch,
    configured_seconds: int | None,
) -> None:
    """Pass default and user-configured seconds to the polling coordinator."""
    options = {} if configured_seconds is None else {CONF_HARDWARE_BATTERY_INTERVAL: configured_seconds}
    connection = SimpleNamespace(open=AsyncMock(), structure_file={})
    hardware_api = SimpleNamespace(async_discover=AsyncMock())
    hardware = SimpleNamespace(
        async_config_entry_first_refresh=AsyncMock(),
        data=SimpleNamespace(devices={}),
    )
    hardware_factory = Mock(return_value=hardware)
    monkeypatch.setattr(coordinator_module, "async_get_clientsession", Mock())
    monkeypatch.setattr(coordinator_module, "LoxoneConnection", Mock(return_value=connection))
    monkeypatch.setattr(coordinator_module, "MiniServer", Mock())
    monkeypatch.setattr(coordinator_module, "LoxoneHardwareApi", Mock(return_value=hardware_api))
    monkeypatch.setattr(coordinator_module, "LoxoneHardwareCoordinator", hardware_factory)
    monkeypatch.setattr(coordinator_module, "configure_hardware_control_registry", Mock())
    coordinator = SimpleNamespace(
        api=None,
        hass=Mock(),
        config_entry=SimpleNamespace(data={}, options=options),
        _host="192.0.2.1",
        _port=80,
        _username="test-user",
        _password="",
        _verify_ssl=True,
        _on_token_changed=Mock(),
    )

    asyncio.run(
        coordinator_module.LoxoneCoordinator.async_config_entry_first_refresh(coordinator),
    )

    hardware_factory.assert_called_once()
    expected_seconds = 900 if configured_seconds is None else configured_seconds
    assert hardware_factory.call_args.kwargs["battery_interval"] == expected_seconds
    assert coordinator.hardware is hardware
    hardware.async_config_entry_first_refresh.assert_awaited_once()
