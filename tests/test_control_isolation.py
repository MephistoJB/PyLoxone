"""Protect the shared LoxAPP source from entity-platform normalization."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.loxone import binary_sensor, helpers, sensor
from custom_components.loxone.hardware.api import LoxoneHardwareApi
from custom_components.loxone.hardware.coordinator import LoxoneHardwareCoordinator

STRUCTURE = {
    "rooms": {"room-1": {"name": "Living room"}},
    "cats": {"cat-1": {"name": "Windows"}},
    "controls": {
        "position": {
            "uuidAction": "position",
            "name": "Handle_Position",
            "type": "InfoOnlyAnalog",
            "room": "room-1",
            "cat": "cat-1",
            "details": {"format": "%.0f"},
            "states": {"value": "position-state"},
        },
        "vibration": {
            "uuidAction": "vibration",
            "name": "Handle_Alarm",
            "type": "InfoOnlyDigital",
            "room": "room-1",
            "cat": "cat-1",
            "states": {"active": "vibration-state"},
        },
    },
}
STATUS = '<Status><AirDevice Type="Fenstergriff Air" Serial="50:4F:94:FF:FE:B2:99:C3" Online="true" /></Status>'
CHANNEL_NAMES = "Handle_Position (0C000001.B299C3.AI2,0), Handle_Alarm (0C000001.B299C3.I4,0)"


@pytest.mark.parametrize("types", ["InfoOnlyAnalog", ["InfoOnlyAnalog", "InfoOnlyDigital"]])
def test_get_all_isolates_top_level_and_nested_fields(types):
    """Both filter forms retain source types, metadata, states and subcontrols."""
    source = deepcopy(STRUCTURE)
    source["controls"]["position"]["subControls"] = {"child": {"states": {"value": "original"}}}
    original = deepcopy(source)
    selected = helpers.get_all(source, types)
    assert [item["uuidAction"] for item in selected] == (
        ["position"] if isinstance(types, str) else ["position", "vibration"]
    )
    control = helpers.add_room_and_cat_to_value_values(source, selected[0])
    control["type"] = "analog"
    control["states"]["value"] = "changed"
    control["subControls"]["child"]["states"]["value"] = "changed"
    assert control["room"] == "Living room"
    assert control["cat"] == "Windows"
    assert source == original
    assert helpers.get_all(source, types)[0] == original["controls"]["position"]


@pytest.mark.parametrize("types", [[], "NoSuchType", ["NoSuchType"]])
def test_get_all_keeps_empty_filter_results(types):
    """Copying must not broaden the existing filter behavior."""
    assert helpers.get_all(STRUCTURE, types) == []


def test_real_platform_setup_preserves_repeated_inventory_and_push_mapping(monkeypatch):
    """Real sensor setup must not cause the registry's mapping-change reload condition."""

    async def exercise():
        source = deepcopy(STRUCTURE)
        original = deepcopy(source)
        entry = SimpleNamespace(entry_id="test-entry", data={}, options={})
        miniserver = SimpleNamespace(
            serial="ABCDEF123456",
            lox_config=SimpleNamespace(json=source),
            listeners=[],
            async_signal_new_device=lambda name: name,
        )
        hass = SimpleNamespace(data={"loxone": {entry.entry_id: SimpleNamespace(hardware=None)}})
        for platform in (sensor, binary_sensor):
            monkeypatch.setattr(platform, "get_miniserver_from_hass", lambda *args: miniserver)
            monkeypatch.setattr(platform, "async_dispatcher_connect", lambda *args: Mock())
        monkeypatch.setattr(helpers, "device_registry", {})
        monkeypatch.setattr(helpers, "hardware_control_registry", {})
        monkeypatch.setattr(helpers, "hardware_control_roles", {})
        api = LoxoneHardwareApi(
            Mock(),
            host="192.0.2.1",
            port=80,
            username="test",
            password="test",
            verify_ssl=True,
            structure=source,
        )

        async def read_json(path):
            values = {
                "jdev/sps/enumdev": "Miniserver (ABCDEF123456), Air Base (ABCDEF123456.0C000001)",
                "jdev/sps/enumin": CHANNEL_NAMES,
                "jdev/sps/enumout": "",
            }
            return {"LL": {"Code": "200", "value": values[path]}}

        monkeypatch.setattr(api, "_request_json", read_json)
        monkeypatch.setattr(api, "_request_text", AsyncMock(return_value=STATUS))
        initial = await api.async_discover()
        known_mappings = dict(initial.devices["B299C3"].resolved_mappings)
        known_push = dict(api.push_mapping)
        assert known_mappings == {"position": "position", "vibration": "vibration"}
        added = []

        def add_entities(entities, update_before_add=False):
            added.extend(entities)

        await sensor.async_setup_entry(hass, entry, add_entities)
        await binary_sensor.async_setup_entry(hass, entry, add_entities)
        assert any(isinstance(entity, sensor.LoxoneSensor) for entity in added)
        assert any(isinstance(entity, binary_sensor.LoxoneDigitalSensor) for entity in added)
        assert source == original

        async def keep_runtime(data):
            return data

        monkeypatch.setattr(api, "async_refresh_fast", keep_runtime)
        monkeypatch.setattr(api, "async_refresh_battery_and_temperature", keep_runtime)
        coordinator = LoxoneHardwareCoordinator.__new__(LoxoneHardwareCoordinator)
        coordinator.api = api
        coordinator.data = initial
        coordinator.hass = SimpleNamespace(bus=SimpleNamespace(async_fire=Mock()))
        coordinator.inventory_interval = 30
        coordinator.battery_interval = 900
        coordinator._last_slow_poll = 0
        coordinator._push_versions = {}
        for _ in range(20):
            coordinator._last_inventory = 0
            current = await coordinator._async_update_data()
            assert set(current.devices) == set(initial.devices)
            assert current.devices["B299C3"].resolved_mappings == known_mappings
            assert api.push_mapping == known_push
            assert current.controls["position"].room == "Living room"
            assert current.controls["position"].category == "Windows"
            coordinator.data = current
        assert source == original

        # Genuine assignment changes remain visible to the unchanged reload logic.
        source["controls"]["position"]["name"] = "Renamed_Position"
        changed = await api.async_discover()
        assert changed.devices["B299C3"].resolved_mappings != known_mappings
        assert changed.devices["B299C3"].resolved_mappings == {"vibration": "vibration"}

    asyncio.run(exercise())
