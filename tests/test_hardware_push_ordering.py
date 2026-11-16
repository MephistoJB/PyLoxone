"""New WebSocket values must survive older in-flight polling snapshots."""

from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.loxone.hardware.api import LoxoneHardwareApi
from custom_components.loxone.hardware.coordinator import LoxoneHardwareCoordinator
from custom_components.loxone.hardware.models import HardwareData, HardwareDevice


def make_coordinator():
    """Use real coordinator methods with isolated HA notification boundaries."""
    data = HardwareData("server", "Home", None, "base", None)
    for device_id in ("A", "B"):
        data.devices[device_id] = HardwareDevice(
            device_id=device_id,
            serial=device_id,
            name=device_id,
            device_type="Fenstergriff Air",
            air_base="base",
            online=True,
            position=3,
            vibration=False,
            battery=50,
            system_temperature=20,
            resolved_mappings={
                "position": f"{device_id.lower()}-position",
                "vibration": f"{device_id.lower()}-vibration",
            },
        )

    async def identity(snapshot):
        return deepcopy(snapshot)

    api = SimpleNamespace(
        push_mapping={
            f"{device_id.lower()}-{role}-state": (device_id, role)
            for device_id in ("A", "B")
            for role in ("position", "vibration")
        },
        async_discover=AsyncMock(return_value=deepcopy(data)),
        async_refresh_fast=AsyncMock(side_effect=identity),
        async_refresh_battery_and_temperature=AsyncMock(side_effect=identity),
    )
    co = LoxoneHardwareCoordinator.__new__(LoxoneHardwareCoordinator)
    co.api = api
    co.data = data
    co.inventory_interval = 30
    co.battery_interval = 900
    co._last_inventory = co._last_slow_poll = time.monotonic()
    co._push_versions = {}
    published = []
    notifications = []
    co.hass = SimpleNamespace(
        bus=SimpleNamespace(async_fire=lambda event, payload: published.append(deepcopy(payload)))
    )

    def commit(snapshot):
        co.data = snapshot
        notifications.append(deepcopy(snapshot))

    co.async_set_updated_data = commit
    return SimpleNamespace(co=co, api=api, published=published, notifications=notifications)


@pytest.mark.parametrize("stage", ["inventory", "fast", "slow"])
@pytest.mark.parametrize("role", ["position", "vibration"])
@pytest.mark.parametrize("unchanged", [False, True])
def test_push_during_polling_wins_only_for_its_device_and_signal(stage, role, unchanged):
    """Cover every await stage, both roles, and repeated equal push confirmations."""

    async def exercise():
        f = make_coordinator()
        entered, release = asyncio.Event(), asyncio.Event()
        initial = getattr(f.co.data.devices["A"], role)
        changed = 1 if role == "position" else True
        incoming = initial if unchanged else changed
        stale = changed if unchanged else initial

        def old_poll(snapshot):
            result = deepcopy(snapshot)
            setattr(result.devices["A"], role, stale)
            other_role = "vibration" if role == "position" else "position"
            setattr(result.devices["A"], other_role, True if other_role == "vibration" else 1)
            result.devices["B"].position = 1
            return result

        async def fast(snapshot):
            if stage == "fast":
                entered.set()
                await release.wait()
            return old_poll(snapshot)

        async def inventory():
            snapshot = deepcopy(f.co.data)
            entered.set()
            await release.wait()
            return snapshot

        async def slow(snapshot):
            entered.set()
            await release.wait()
            result = deepcopy(snapshot)
            result.devices["A"].battery = 75
            result.devices["A"].system_temperature = 42
            return result

        f.api.async_refresh_fast.side_effect = fast
        if stage == "inventory":
            f.co._last_inventory = 0
            f.api.async_discover.side_effect = inventory
        if stage == "slow":
            f.co._last_slow_poll = 0
            f.api.async_refresh_battery_and_temperature.side_effect = slow
        task = asyncio.create_task(f.co._async_update_data())
        await entered.wait()
        try:
            await f.co.async_apply_push({f"a-{role}-state": incoming})
            assert getattr(f.co.data.devices["A"], role) == incoming
            assert f.co._push_versions[("A", role)] == 1
            timestamp = f.co.data.last_push
        finally:
            release.set()
        result = await task
        assert getattr(result.devices["A"], role) == incoming
        assert result.last_push == timestamp
        other_role = "vibration" if role == "position" else "position"
        assert getattr(result.devices["A"], other_role) == (True if other_role == "vibration" else 1)
        assert result.devices["B"].position == 1
        assert f.published[-1][f"a-{role}"] == float(incoming)
        if stage == "slow":
            assert result.devices["A"].battery == 75
            assert result.devices["A"].system_temperature == 42

        # With no newer push during the next cycle, polling still provides fallback.
        f.co.data = result
        f.co._last_inventory = f.co._last_slow_poll = time.monotonic()
        next_result = await f.co._async_update_data()
        assert getattr(next_result.devices["A"], role) == stale

    asyncio.run(exercise())


def test_actual_channel_poll_cannot_roll_back_latest_of_multiple_pushes():
    """Use the real channel-poll implementation and check synthetic entity events."""

    async def exercise():
        f = make_coordinator()
        api = LoxoneHardwareApi(
            Mock(), host="192.0.2.1", port=80, username="test", password="test", verify_ssl=True, structure={}
        )
        api.push_mapping = f.api.push_mapping
        f.co.api = api
        entered, release = asyncio.Event(), asyncio.Event()

        async def read_channel(device, channel):
            if device.device_id == "A" and channel == "AI2":
                entered.set()
                await release.wait()
                return "3"
            return "0" if channel == "I4" else "1"

        api.async_read_channel = read_channel
        task = asyncio.create_task(f.co._async_update_data())
        await entered.wait()
        try:
            await f.co.async_apply_push({"a-position-state": 2})
            await f.co.async_apply_push({"a-position-state": 1, "a-vibration-state": 1})
        finally:
            release.set()
        result = await task
        assert result.devices["A"].position == 1
        assert result.devices["A"].vibration is True
        assert result.devices["B"].position == 1
        assert f.co._push_versions == {("A", "position"): 2, ("A", "vibration"): 1}
        assert f.published == [{"a-position": 1.0, "a-vibration": 1.0, "b-position": 1.0, "b-vibration": 0.0}]

    asyncio.run(exercise())


def test_cancelled_poll_keeps_push_and_emits_no_old_fallback():
    """Cancellation must not undo a push or synthesize stale entity events."""

    async def exercise():
        f = make_coordinator()
        entered = asyncio.Event()

        async def blocked(snapshot):
            entered.set()
            await asyncio.Event().wait()

        f.api.async_refresh_fast.side_effect = blocked
        task = asyncio.create_task(f.co._async_update_data())
        await entered.wait()
        await f.co.async_apply_push({"a-position-state": 1})
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert f.co.data.devices["A"].position == 1
        assert f.published == []

    asyncio.run(exercise())


def test_inventory_removal_does_not_resurrect_a_pushed_device():
    """A newer push must not hide a genuine removal from hardware inventory."""

    async def exercise():
        f = make_coordinator()
        f.co._last_inventory = 0
        entered, release = asyncio.Event(), asyncio.Event()

        async def inventory():
            snapshot = deepcopy(f.co.data)
            snapshot.devices.pop("A")
            entered.set()
            await release.wait()
            return snapshot

        f.api.async_discover.side_effect = inventory
        task = asyncio.create_task(f.co._async_update_data())
        await entered.wait()
        try:
            await f.co.async_apply_push({"a-position-state": 1})
        finally:
            release.set()
        result = await task
        assert set(result.devices) == {"B"}
        assert "a-position" not in f.published[-1]

    asyncio.run(exercise())


def test_unmapped_or_missing_devices_do_not_advance_versions():
    """Only mapped incoming values for present devices affect poll ordering."""

    async def exercise():
        f = make_coordinator()
        f.co.data.devices.pop("A")
        await f.co.async_apply_push({"unmapped-state": 1, "a-position-state": 1})
        assert f.co._push_versions == {}
        assert f.notifications == []

    asyncio.run(exercise())
