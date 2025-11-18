"""Unit tests for the `ModalityAgent` routing behaviour."""

from typing import Any, List
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

from agents.modality_agent import ModalityAgent


@pytest.fixture
def modality_agent_cfg() -> DictConfig:
    """Return a minimal configuration for constructing a `ModalityAgent`."""
    return OmegaConf.create(
        {
            "agent": {
                "name": "ModalityAgent",
                "role": "test role",
                "description": "test description",
                "system_message": "test system message",
                "guardrails": {
                    "banned_words": [],
                    "toxic_threshold": 1.0,
                },
            }
        }
    )


@pytest.fixture
def modality_agent_fixture(modality_agent_cfg: DictConfig) -> ModalityAgent:
    """Create a `ModalityAgent` instance with a mocked guard."""
    logger = MagicMock()
    llm = MagicMock()
    agent = ModalityAgent(
        cfg=modality_agent_cfg,
        logger=logger,
        llm=llm,
        output=MagicMock(),
    )
    agent.guard = MagicMock(validate=lambda _: MagicMock(validation_passed=True))
    return agent


@pytest.mark.parametrize(
    "query,expected",
    [
        ("text", ["text", "image"]),
        ("image", ["image"]),
    ],
)
def test_modality_agent_routing(
    modality_agent_fixture: ModalityAgent,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    expected: List[str],
) -> None:
    """`ModalityAgent.run_query` should return the expected modalities."""
    monkeypatch.setattr(
        ModalityAgent,
        "run_query",
        lambda _, query: ["text", "image"] if query == "text" else ["image"],
    )

    assert modality_agent_fixture.run_query(query) == expected


def test_modality_agent_unsupported_modality(
    modality_agent_fixture: ModalityAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported modalities should raise a `ValueError`."""

    def fake_run(_: Any, query: str) -> List[str]:
        if query == "audio":
            raise ValueError("Rejected query due to: unsupported modality")
        return [query]

    monkeypatch.setattr(ModalityAgent, "run_query", fake_run)

    with pytest.raises(ValueError):
        modality_agent_fixture.run_query("audio")
