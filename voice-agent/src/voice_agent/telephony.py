from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import ssl

import aiohttp
import certifi
from livekit import api

from voice_agent.settings import AppSettings


@dataclass(frozen=True, slots=True)
class SIPInboundTrunkSummary:
    trunk_id: str
    name: str
    numbers: list[str]


@dataclass(frozen=True, slots=True)
class SIPDispatchRuleSummary:
    rule_id: str
    name: str
    trunk_ids: list[str]
    room_prefix: str
    agent_names: list[str]


@dataclass(frozen=True, slots=True)
class SIPTelephonyStatus:
    phone_number: str | None
    sip_domain: str | None
    inbound_trunk: SIPInboundTrunkSummary | None
    dispatch_rule: SIPDispatchRuleSummary | None


@dataclass(frozen=True, slots=True)
class SIPProvisionResult:
    status: SIPTelephonyStatus
    created_inbound_trunk: bool
    created_dispatch_rule: bool


class LiveKitTelephonyManager:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    async def fetch_status(self) -> SIPTelephonyStatus:
        async with self._livekit_api() as lkapi:
            trunks = await lkapi.sip.list_inbound_trunk(api.ListSIPInboundTrunkRequest())
            rules = await lkapi.sip.list_dispatch_rule(api.ListSIPDispatchRuleRequest())

        inbound_trunk = self._match_inbound_trunk(trunks.items)
        dispatch_rule = self._match_dispatch_rule(
            rules.items,
            trunk_id=inbound_trunk.trunk_id if inbound_trunk else None,
        )
        return self._build_status(inbound_trunk, dispatch_rule)

    async def provision(self) -> SIPProvisionResult:
        phone_number = self._settings.twilio_phone_number
        if not phone_number:
            raise ValueError("TWILIO_PHONE_NUMBER is required to provision SIP resources")

        async with self._livekit_api() as lkapi:
            trunks = await lkapi.sip.list_inbound_trunk(api.ListSIPInboundTrunkRequest())
            inbound_trunk = self._match_inbound_trunk(trunks.items)
            created_inbound_trunk = False
            if inbound_trunk is None:
                created = await lkapi.sip.create_inbound_trunk(
                    api.CreateSIPInboundTrunkRequest(
                        trunk=api.SIPInboundTrunkInfo(
                            name=self._settings.sip_inbound_trunk_name,
                            numbers=[phone_number],
                            metadata=f"agent={self._settings.agent_name}",
                        )
                    )
                )
                inbound_trunk = self._summarize_inbound_trunk(created)
                created_inbound_trunk = True

            rules = await lkapi.sip.list_dispatch_rule(api.ListSIPDispatchRuleRequest())
            dispatch_rule = self._match_dispatch_rule(
                rules.items,
                trunk_id=inbound_trunk.trunk_id,
            )
            created_dispatch_rule = False
            if dispatch_rule is None:
                created = await lkapi.sip.create_dispatch_rule(
                    api.CreateSIPDispatchRuleRequest(
                        rule=api.SIPDispatchRule(
                            dispatch_rule_individual=api.SIPDispatchRuleIndividual(
                                room_prefix=self._settings.sip_room_prefix,
                            )
                        ),
                        trunk_ids=[inbound_trunk.trunk_id],
                        name=self._settings.sip_dispatch_rule_name,
                        metadata=f"agent={self._settings.agent_name}",
                        room_config=api.RoomConfiguration(
                            agents=[
                                api.RoomAgentDispatch(
                                    agent_name=self._settings.agent_name
                                )
                            ]
                        ),
                    )
                )
                dispatch_rule = self._summarize_dispatch_rule(created)
                created_dispatch_rule = True

        return SIPProvisionResult(
            status=self._build_status(inbound_trunk, dispatch_rule),
            created_inbound_trunk=created_inbound_trunk,
            created_dispatch_rule=created_dispatch_rule,
        )

    @asynccontextmanager
    async def _livekit_api(self):
        connector = aiohttp.TCPConnector(
            ssl=ssl.create_default_context(cafile=certifi.where())
        )
        session = aiohttp.ClientSession(connector=connector)
        lkapi = api.LiveKitAPI(
            url=self._settings.livekit_url,
            api_key=self._settings.livekit_api_key,
            api_secret=self._settings.livekit_api_secret,
            session=session,
        )
        try:
            yield lkapi
        finally:
            await session.close()

    def _build_status(
        self,
        inbound_trunk: SIPInboundTrunkSummary | None,
        dispatch_rule: SIPDispatchRuleSummary | None,
    ) -> SIPTelephonyStatus:
        return SIPTelephonyStatus(
            phone_number=self._settings.twilio_phone_number,
            sip_domain=self._settings.twilio_sip_domain,
            inbound_trunk=inbound_trunk,
            dispatch_rule=dispatch_rule,
        )

    def _match_inbound_trunk(
        self,
        trunks: list[api.SIPInboundTrunkInfo],
    ) -> SIPInboundTrunkSummary | None:
        configured_number = self._settings.twilio_phone_number
        for trunk in trunks:
            summary = self._summarize_inbound_trunk(trunk)
            if summary.name == self._settings.sip_inbound_trunk_name:
                return summary
            if configured_number and configured_number in summary.numbers:
                return summary
        return None

    def _match_dispatch_rule(
        self,
        rules: list[api.SIPDispatchRuleInfo],
        *,
        trunk_id: str | None,
    ) -> SIPDispatchRuleSummary | None:
        for rule in rules:
            summary = self._summarize_dispatch_rule(rule)
            if summary.name == self._settings.sip_dispatch_rule_name:
                return summary
            if trunk_id and trunk_id in summary.trunk_ids:
                return summary
            if self._settings.agent_name in summary.agent_names:
                return summary
        return None

    @staticmethod
    def _summarize_inbound_trunk(
        trunk: api.SIPInboundTrunkInfo,
    ) -> SIPInboundTrunkSummary:
        return SIPInboundTrunkSummary(
            trunk_id=trunk.sip_trunk_id,
            name=trunk.name,
            numbers=[str(number) for number in trunk.numbers],
        )

    @staticmethod
    def _summarize_dispatch_rule(
        rule: api.SIPDispatchRuleInfo,
    ) -> SIPDispatchRuleSummary:
        room_prefix = ""
        if rule.rule.HasField("dispatch_rule_individual"):
            room_prefix = rule.rule.dispatch_rule_individual.room_prefix
        agent_names = [
            agent.agent_name
            for agent in getattr(rule.room_config, "agents", [])
            if agent.agent_name
        ]
        return SIPDispatchRuleSummary(
            rule_id=rule.sip_dispatch_rule_id,
            name=rule.name,
            trunk_ids=[str(value) for value in rule.trunk_ids],
            room_prefix=room_prefix,
            agent_names=agent_names,
        )
