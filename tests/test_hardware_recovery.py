"""Recovery of optional hardware must not tear down a healthy main connection."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from homeassistant.exceptions import ConfigEntryNotReady

from custom_components.loxone import coordinator as coordinator_module
from custom_components.loxone.const import CONF_HARDWARE_ENABLED, CONF_HARDWARE_MAPPINGS
from custom_components.loxone.hardware.api import LoxoneHardwareError


@pytest.fixture
def recovery(monkeypatch):
    """Use the real setup/recovery methods with isolated HTTP and HA boundaries."""
    tasks = []

    def background_task(hass, coroutine, *, name):
        task = asyncio.create_task(coroutine, name=name)
        tasks.append(task)
        return task

    entry = SimpleNamespace(
        entry_id="test-miniserver",
        title="Test Miniserver",
        options={CONF_HARDWARE_ENABLED: True},
        data={CONF_HARDWARE_MAPPINGS: {"serial:position": "control-uuid"}},
        async_create_background_task=background_task,
    )
    main = SimpleNamespace(
        open=AsyncMock(),
        close=AsyncMock(),
        structure_file={},
        is_connected=True,
    )
    hardware_api = SimpleNamespace(async_discover=AsyncMock())
    hardware = SimpleNamespace(
        async_config_entry_first_refresh=AsyncMock(),
        async_refresh=AsyncMock(),
        last_update_success=True,
        last_exception=None,
        async_shutdown=AsyncMock(),
        data=SimpleNamespace(devices={}),
    )
    factory = Mock(return_value=hardware)
    monkeypatch.setattr(coordinator_module, "async_get_clientsession", Mock())
    monkeypatch.setattr(coordinator_module, "LoxoneConnection", Mock(return_value=main))
    monkeypatch.setattr(coordinator_module, "MiniServer", Mock())
    monkeypatch.setattr(coordinator_module, "LoxoneHardwareApi", Mock(return_value=hardware_api))
    monkeypatch.setattr(coordinator_module, "LoxoneHardwareCoordinator", factory)
    mapping = Mock()
    monkeypatch.setattr(coordinator_module, "configure_hardware_control_registry", mapping)
    coordinator = coordinator_module.LoxoneCoordinator.__new__(coordinator_module.LoxoneCoordinator)
    coordinator.__dict__.update(
        api=None,
        hardware=None,
        listeners=[],
        config_entry=entry,
        hass=SimpleNamespace(
            async_add_executor_job=Mock(),
            config_entries=SimpleNamespace(async_schedule_reload=Mock()),
        ),
        _host="192.0.2.1",
        _port=80,
        _username="test",
        _password="test",
        _verify_ssl=True,
        _hardware_retry_api=None,
        _hardware_retry_task=None,
        _closing=False,
    )
    return SimpleNamespace(
        coordinator=coordinator,
        main=main,
        api=hardware_api,
        hardware=hardware,
        factory=factory,
        tasks=tasks,
        mapping=mapping,
        reload=coordinator.hass.config_entries.async_schedule_reload,
    )


async def test_startup_401_recovers_with_one_reload(recovery, monkeypatch):
    """A failed startup stays usable and discovers hardware later without manual action."""
    co = recovery.coordinator
    recovery.api.async_discover.side_effect = [LoxoneHardwareError("401 Unauthorized"), object()]
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(coordinator_module.asyncio, "sleep", sleep)
    await co.async_config_entry_first_refresh()
    assert co.api is recovery.main
    assert co.hardware is None
    assert recovery.tasks == []  # Do not start retries during platform setup.
    recovery.reload.assert_not_called()
    recovery.main.close.assert_not_awaited()

    co.async_start_hardware_recovery()
    co.async_start_hardware_recovery()  # The entry can only own one retry loop.
    assert len(recovery.tasks) == 1
    await recovery.tasks[0]

    assert delays == [30]
    assert recovery.api.async_discover.await_count == 2
    recovery.hardware.async_config_entry_first_refresh.assert_not_awaited()
    recovery.hardware.async_refresh.assert_awaited_once()
    recovery.hardware.async_shutdown.assert_awaited_once()
    recovery.reload.assert_called_once_with("test-miniserver")
    recovery.main.close.assert_not_awaited()
    assert co._hardware_retry_api is None
    # The probe must not replace mappings/entities under running platforms.
    recovery.mapping.assert_called_once_with({})
    co.async_start_hardware_recovery()
    assert len(recovery.tasks) == 1


async def test_persistent_failure_uses_capped_backoff(recovery, monkeypatch):
    """Repeated rejection never causes reload storms or a busy retry loop."""
    co = recovery.coordinator
    recovery.api.async_discover.side_effect = [LoxoneHardwareError("401")] * 8 + [object()]
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(coordinator_module.asyncio, "sleep", sleep)
    await co.async_config_entry_first_refresh()
    co.async_start_hardware_recovery()
    await recovery.tasks[0]

    assert delays == [30, 60, 120, 240, 300, 300, 300, 300]
    recovery.reload.assert_called_once()
    recovery.main.close.assert_not_awaited()


async def test_first_poll_failure_is_cleaned_up_and_retried(recovery, monkeypatch):
    """Successful inventory alone is not recovery when the first poll still fails."""
    recovery.hardware.async_config_entry_first_refresh.side_effect = ConfigEntryNotReady(
        "temporary hardware poll failure"
    )
    poll_results = iter([False, True])

    def poll():
        recovery.hardware.last_update_success = next(poll_results)
        recovery.hardware.last_exception = LoxoneHardwareError("still failing")

    recovery.hardware.async_refresh.side_effect = poll
    monkeypatch.setattr(coordinator_module.asyncio, "sleep", AsyncMock())
    co = recovery.coordinator
    await co.async_config_entry_first_refresh()
    assert co.hardware is None
    assert recovery.hardware.async_shutdown.await_count == 1
    co.async_start_hardware_recovery()
    await recovery.tasks[0]

    recovery.hardware.async_config_entry_first_refresh.assert_awaited_once()
    assert recovery.hardware.async_refresh.await_count == 2
    assert recovery.hardware.async_shutdown.await_count == 3
    recovery.reload.assert_called_once()
    recovery.main.close.assert_not_awaited()


@pytest.mark.parametrize("during_request", [False, True])
async def test_unload_cancels_recovery_without_reviving_entry(recovery, monkeypatch, during_request):
    """Unload/disable cancels both backoff and in-flight hardware discovery."""
    co = recovery.coordinator
    recovery.api.async_discover.side_effect = LoxoneHardwareError("401")
    await co.async_config_entry_first_refresh()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked(*args):
        entered.set()
        await release.wait()

    if during_request:
        monkeypatch.setattr(coordinator_module.asyncio, "sleep", AsyncMock())
        recovery.api.async_discover.side_effect = blocked
    else:
        monkeypatch.setattr(coordinator_module.asyncio, "sleep", blocked)
    co.async_start_hardware_recovery()
    task = recovery.tasks[0]
    await entered.wait()
    await co.async_cleanup()

    assert task.cancelled()
    recovery.reload.assert_not_called()
    recovery.main.close.assert_awaited_once()
    co.async_start_hardware_recovery()
    assert len(recovery.tasks) == 1


async def test_successful_or_disabled_hardware_does_not_start_recovery(recovery):
    """Healthy and explicitly disabled hardware produce no background probes."""
    co = recovery.coordinator
    await co.async_config_entry_first_refresh()
    assert co.hardware is recovery.hardware
    co.async_start_hardware_recovery()
    assert recovery.tasks == []
    await co.async_cleanup()

    recovery.coordinator._closing = False
    recovery.coordinator.hardware = None
    recovery.coordinator.api = None
    recovery.coordinator.config_entry.options[CONF_HARDWARE_ENABLED] = False
    await co.async_config_entry_first_refresh()
    co.async_start_hardware_recovery()
    assert recovery.tasks == []
    assert recovery.api.async_discover.await_count == 1


async def test_real_http_discovery_recovers_and_rebuilds_entities(hass, aioclient_mock, monkeypatch):
    """Exercise aiohttp, HA entry tasks and the real hardware polling coordinator."""
    from homeassistant.config_entries import ConfigEntryState
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    from custom_components.loxone.hardware.entity import hardware_sensor_entities

    entry = MockConfigEntry(
        domain="loxone",
        version=6,
        options={"host": "192.0.2.1", "port": 80, "username": "test", "password": ""},
    )
    entry.add_to_hass(hass)
    entry.mock_state(hass, ConfigEntryState.SETUP_IN_PROGRESS)
    main = SimpleNamespace(open=AsyncMock(), close=AsyncMock(), connection=None, structure_file={}, is_connected=True)
    monkeypatch.setattr(coordinator_module, "LoxoneConnection", Mock(return_value=main))
    monkeypatch.setattr(coordinator_module, "MiniServer", Mock())
    scheduled_reload = Mock()
    monkeypatch.setattr(hass.config_entries, "async_schedule_reload", scheduled_reload)
    base = "http://192.0.2.1"
    status = """<Status>
      <Miniserver Name="Test Home" Version="16.1" />
      <Extension Type="Air Base Extension" Serial="0C000001" />
      <AirDevice Type="Fenstergriff Air" Serial="50:4F:94:FF:FE:B2:99:C3"
        Name="EG_WZ_Fenster" Online="true" Battery="70%" />
    </Status>"""

    def responses(*, rejected):
        aioclient_mock.get(
            base + "/jdev/sps/enumdev",
            json={"LL": {"Code": "200", "value": "Miniserver (ABCDEF123456), Air Base (ABCDEF123456.0C000001)"}},
        )
        aioclient_mock.get(base + "/jdev/sps/enumin", json={"LL": {"Code": "200", "value": ""}})
        if rejected:
            aioclient_mock.get(base + "/jdev/sps/enumout", status=401)
        else:
            aioclient_mock.get(base + "/jdev/sps/enumout", json={"LL": {"Code": "200", "value": ""}})
        aioclient_mock.get(base + "/data/status", text=status)
        for channel, value in {"AI2": "1", "I4": "0", "ST": "25", "AI0": "70", "I0": "0"}.items():
            aioclient_mock.get(
                base + "/jdev/sps/io/0C000001.B299C3." + channel, json={"LL": {"Code": "200", "value": value}}
            )

    responses(rejected=True)
    co = coordinator_module.LoxoneCoordinator(hass, entry)
    await co.async_config_entry_first_refresh()
    assert co.hardware is None
    main.close.assert_not_awaited()

    entry.mock_state(hass, ConfigEntryState.LOADED)
    aioclient_mock.clear_requests()
    responses(rejected=False)
    # Skip wall-clock backoff, while retaining the actual HTTP/polling and HA tasks.
    retries = []

    async def skip_backoff(delay):
        retries.append(delay)
        assert len(retries) == 1, "Real HTTP recovery failed on its first probe"

    with pytest.MonkeyPatch.context() as clock:
        clock.setattr(coordinator_module.asyncio, "sleep", skip_backoff)
        co.async_start_hardware_recovery()
        await co._hardware_retry_task
    scheduled_reload.assert_called_once_with(entry.entry_id)
    main.close.assert_not_awaited()
    await co.async_cleanup()

    # The scheduled reload goes through normal setup with the same registry identity.
    entry.mock_state(hass, ConfigEntryState.SETUP_IN_PROGRESS)
    rebuilt = coordinator_module.LoxoneCoordinator(hass, entry)
    await rebuilt.async_config_entry_first_refresh()
    assert rebuilt.hardware is not None
    device = rebuilt.hardware.data.devices["B299C3"]
    assert device.online is True
    assert device.battery == 70
    assert device.position_name == "closed"
    entities = hardware_sensor_entities(rebuilt.hardware, entry)
    assert any(entity.unique_id.endswith("-battery") for entity in entities)
    await rebuilt.async_cleanup()
    entry.mock_state(hass, ConfigEntryState.NOT_LOADED)
