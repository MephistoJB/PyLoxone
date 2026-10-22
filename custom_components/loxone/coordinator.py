import asyncio
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PASSWORD, CONF_PORT, CONF_USERNAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import (
    CONF_HARDWARE_BATTERY_INTERVAL,
    CONF_HARDWARE_ENABLED,
    CONF_HARDWARE_FAST_POLL_INTERVAL,
    CONF_HARDWARE_INVENTORY_INTERVAL,
    CONF_HARDWARE_MAPPINGS,
    CONF_VERIFY_SSL,
    DEFAULT_HARDWARE_BATTERY_INTERVAL,
    DEFAULT_HARDWARE_ENABLED,
    DEFAULT_HARDWARE_FAST_POLL_INTERVAL,
    DEFAULT_HARDWARE_INVENTORY_INTERVAL,
    DEFAULT_VERIFY_SSL,
)
from .hardware import LoxoneHardwareApi, LoxoneHardwareCoordinator
from .hardware.api import LoxoneHardwareError
from .hardware.entity import control_device_mapping
from .helpers import configure_hardware_control_registry
from .miniserver import MiniServer
from .pyloxone_api.connection import LoxoneConnection, LoxoneException

_LOGGER = logging.getLogger(__name__)

HARDWARE_RETRY_DELAY = 30
HARDWARE_RETRY_MAX_DELAY = 300


def _collect_control_uuids(structure_file: dict) -> frozenset[str]:
    """Collect the UUIDs of all controls and subcontrols (``uuidAction``)."""
    uuids: set[str] = set()
    for key, control in (structure_file.get("controls") or {}).items():
        uuids.add(key)
        if not isinstance(control, dict):
            continue
        if control.get("uuidAction"):
            uuids.add(control["uuidAction"])
        for sub_key, sub in (control.get("subControls") or {}).items():
            uuids.add(sub_key)
            if isinstance(sub, dict) and sub.get("uuidAction"):
                uuids.add(sub["uuidAction"])
    return frozenset(uuids)


class LoxoneCoordinator(DataUpdateCoordinator):
    """Class to manage fetching data from the Loxone Miniserver."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            logger=_LOGGER,
            config_entry=config_entry,
            name="PyLoxone Coordinator",
            update_method=None,  # Not polling!
        )
        self._username = config_entry.options[CONF_USERNAME]
        self._password = config_entry.options[CONF_PASSWORD]
        self._host = config_entry.options[CONF_HOST]
        self._port = config_entry.options[CONF_PORT]
        self._verify_ssl = config_entry.options.get(CONF_VERIFY_SSL, DEFAULT_VERIFY_SSL)

        self.api: LoxoneConnection | None = None
        self.miniserver: MiniServer | None = None
        self.listeners = []
        self.hardware: LoxoneHardwareCoordinator | None = None
        self._hardware_retry_api: LoxoneHardwareApi | None = None
        self._hardware_retry_task: asyncio.Task | None = None
        self._closing = False
        # Control and subcontrol UUIDs from the structure file; filled
        # by ``async_config_entry_first_refresh``. See ``owns_uuid``.
        self._control_uuids: frozenset[str] = frozenset()
        # Supervisor task running ``LoxoneConnection.run()``. Stored
        # here (instead of a local in ``start_event``) so
        # ``async_unload_entry`` can cancel it — this aborts the
        # backoff sleep between retries, which ``api.close()`` alone
        # would not interrupt.
        self._listening_task: asyncio.Task | None = None
        # False while the websocket connection is down. Entity
        # availability consults this (see ``LoxoneEntity.available``);
        # it is flipped by the ``run()`` supervisor's on_state
        # callback in ``__init__.py``.
        self.connection_available: bool = True

    def _on_token_changed(self, token: dict) -> None:
        """Persist a freshly-issued token into the ConfigEntry.

        Called by :class:`LoxoneConnection` via the
        ``token_change_callback`` whenever the Miniserver issues
        a new token (initial auth or refresh). The token dict
        is merged into ``ConfigEntry.data`` so a HA restart
        can pick it up without re-prompting for credentials.

        Errors are logged at WARNING — a persistence failure
        must not break the live websocket path; the user just
        has to re-authenticate on the next restart.
        """
        try:
            new_data = {**self.config_entry.data, **token}
            # ``async_update_entry`` is the supported way to
            # mutate a ConfigEntry's data from a coordinator.
            self.hass.config_entries.async_update_entry(self.config_entry, data=new_data)
            _LOGGER.info(
                "Persisted updated Loxone token for config entry %s (valid until %s)",
                self.config_entry.entry_id,
                token.get("valid_until", "unknown"),
            )
        except Exception as e:
            _LOGGER.warning("Failed to persist updated token: %s", e, exc_info=True)

    async def async_config_entry_first_refresh(self) -> None:
        _LOGGER.debug("async_config_entry_first_refresh")
        if self.api and self.api.connection:
            await self.api.close()
            self.api.connection = None

        if "token" in self.config_entry.data:
            self.api = LoxoneConnection(
                host=self._host,
                port=self._port,
                username=self._username,
                password=self._password,
                token=self.config_entry.data,
                verify_ssl=self._verify_ssl,
                # Off-load the (potentially multi-MB)
                # ``LoxAPP3.json`` parse to HA's default executor —
                # keeps the event loop responsive while ``open()``
                # is bootstrapping.
                executor=self.hass.async_add_executor_job,
                # ``_on_token_changed`` is async-aware and
                # merges the new token into ConfigEntry.data
                # so a HA restart can resume without a fresh
                # username/password auth.
                token_change_callback=self._on_token_changed,
            )
        else:
            self.api = LoxoneConnection(
                host=self._host,
                port=self._port,
                username=self._username,
                password=self._password,
                verify_ssl=self._verify_ssl,
                executor=self.hass.async_add_executor_job,
                token_change_callback=self._on_token_changed,
            )
        try:
            session = async_get_clientsession(self.hass)
            # Bootstrap only: the websocket is opened by ``run()`` once
            # the platforms are set up. Opened here, it would sit
            # unauthenticated during platform setup and the Miniserver
            # would close it (issue #527).
            await self.api.open(session, connect_websocket=False)
        except LoxoneException as e:
            _LOGGER.error("Could not connect to Loxone Miniserver")
            raise e
        except Exception as e:
            _LOGGER.error("Could not connect to Loxone Miniserver")
            raise e

        self.miniserver = MiniServer(self.hass, self.api.structure_file, self.config_entry)
        self._control_uuids = _collect_control_uuids(self.api.structure_file)

        configure_hardware_control_registry({})
        if self.config_entry.options.get(CONF_HARDWARE_ENABLED, DEFAULT_HARDWARE_ENABLED):
            hardware_api = LoxoneHardwareApi(
                session=session,
                host=self._host,
                port=self._port,
                username=self._username,
                password=self._password,
                verify_ssl=self._verify_ssl,
                structure=self.api.structure_file,
                manual_mappings=self.config_entry.data.get(CONF_HARDWARE_MAPPINGS, {}),
            )
            try:
                self.hardware = await self._async_initialize_hardware(hardware_api)
                configure_hardware_control_registry(control_device_mapping(self.hardware.data))
            except Exception as err:
                # Keep the main connection usable. Start recovery only after
                # the entry's platforms and listeners have finished loading.
                _LOGGER.warning(
                    "Physical hardware inventory unavailable: %s. Retrying in %s s",
                    err,
                    HARDWARE_RETRY_DELAY,
                )
                self.hardware = None
                self._hardware_retry_api = hardware_api

        return None

    async def _async_initialize_hardware(
        self, hardware_api: LoxoneHardwareApi, *, initial_setup: bool = True
    ) -> LoxoneHardwareCoordinator:
        """Discover hardware and verify its first polling cycle."""
        hardware_data = await hardware_api.async_discover()
        hardware = LoxoneHardwareCoordinator(
            self.hass,
            self.config_entry,
            hardware_api,
            hardware_data,
            fast_interval=int(
                self.config_entry.options.get(CONF_HARDWARE_FAST_POLL_INTERVAL, DEFAULT_HARDWARE_FAST_POLL_INTERVAL)
            ),
            inventory_interval=int(
                self.config_entry.options.get(CONF_HARDWARE_INVENTORY_INTERVAL, DEFAULT_HARDWARE_INVENTORY_INTERVAL)
            ),
            battery_interval=int(
                self.config_entry.options.get(CONF_HARDWARE_BATTERY_INTERVAL, DEFAULT_HARDWARE_BATTERY_INTERVAL)
            ),
        )
        hardware.push_connected = lambda: bool(self.api and self.api.is_connected)
        try:
            if initial_setup:
                await hardware.async_config_entry_first_refresh()
            else:
                # HA only permits first_refresh while SETUP_IN_PROGRESS. A
                # recovery probe runs after the main entry is already loaded.
                await hardware.async_refresh()
                if not hardware.last_update_success:
                    raise LoxoneHardwareError("Hardware polling has not recovered") from hardware.last_exception
        except BaseException:
            await hardware.async_shutdown()
            raise
        return hardware

    def async_start_hardware_recovery(self) -> None:
        """Retry optional hardware setup without interrupting the main connection."""
        if self._closing or self._hardware_retry_api is None:
            return
        if self._hardware_retry_task is not None and not self._hardware_retry_task.done():
            return
        self._hardware_retry_task = self.config_entry.async_create_background_task(
            self.hass,
            self._async_retry_hardware(),
            name=f"loxone hardware recovery {self.config_entry.entry_id}",
        )

    async def _async_retry_hardware(self) -> None:
        """Probe with capped backoff and reload this entry once hardware recovers."""
        hardware_api = self._hardware_retry_api
        if hardware_api is None:
            return
        delay = HARDWARE_RETRY_DELAY
        while not self._closing:
            await asyncio.sleep(delay)
            if self._closing:
                return
            try:
                hardware = await self._async_initialize_hardware(hardware_api, initial_setup=False)
            except Exception as err:
                delay = min(delay * 2, HARDWARE_RETRY_MAX_DELAY)
                _LOGGER.debug("Hardware recovery failed: %s. Retrying in %s s", err, delay)
                continue
            # No entities listen to this probe coordinator. Dispose of it before
            # scheduling a reload, which rebuilds all platforms and mappings.
            await hardware.async_shutdown()
            if self._closing:
                return
            self._hardware_retry_api = None
            _LOGGER.info("Physical hardware access restored; reloading %s", self.config_entry.title)
            self.hass.config_entries.async_schedule_reload(self.config_entry.entry_id)
            return

    def owns_uuid(self, uuid: str) -> bool:
        """Return True if ``uuid`` is a control or subcontrol of this Miniserver.

        Used to route commands when more than one Miniserver is set up:
        services and the send events only carry the control UUID.
        """
        return uuid in self._control_uuids

    async def _async_update_data(self) -> None:
        """Fetch data from API endpoint.

        This is the place to pre-process the data to lookup tables
        so entities can quickly look up their data.
        """
        print("_async_update_data")
        return None

    async def async_cleanup(self, kill_token: bool = False) -> None:
        """Clean up resources.

        ``kill_token`` defaults to ``False``: an unload (reload, HA
        restart) must keep the token valid so the value persisted by
        ``_on_token_changed`` survives. Only removing the config entry
        for good invalidates it on the Miniserver (see
        ``async_remove_entry`` in ``__init__.py``).
        """
        self._closing = True
        retry_task = getattr(self, "_hardware_retry_task", None)
        if retry_task is not None and not retry_task.done():
            retry_task.cancel()
            try:
                await retry_task
            except asyncio.CancelledError:
                pass
        self._hardware_retry_task = None
        self._hardware_retry_api = None

        if hasattr(self, "listeners"):
            # Clean up all event listeners. One bad unsubscribe must
            # not prevent the rest (or api.close() below) from running.
            for listener in self.listeners:
                if listener is not None:
                    try:
                        listener()
                    except Exception as err:  # noqa: BLE001
                        _LOGGER.warning(
                            "Failed to unsubscribe Loxone listener %r: %s",
                            listener,
                            err,
                            exc_info=True,
                        )
            self.listeners = []

        if self.hardware is not None:
            try:
                await self.hardware.async_shutdown()
            except Exception as err:
                _LOGGER.warning("Hardware shutdown failed: %s", err, exc_info=True)

        # Close API connection.
        if hasattr(self, "api") and self.api is not None:
            if kill_token:
                # ``kill_token`` invalidates the JWT on the Miniserver
                # so it cannot be reused. Only wanted when the entry is
                # removed for good — otherwise the just-persisted token
                # would be dead on the next start and every restart
                # would have to fall back to username/password auth.
                try:
                    await self.api.kill_token()
                except Exception as err:  # noqa: BLE001
                    _LOGGER.debug("kill_token during cleanup failed: %s", err, exc_info=True)
            try:
                await self.api.close()
            except Exception as err:  # noqa: BLE001
                _LOGGER.warning("api.close during cleanup failed: %s", err, exc_info=True)
