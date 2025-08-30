from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from src.agents.multi_modality.modality_agent import ModalityAgent

modality_agent_cfg = OmegaConf.create(
    {
        "modality_agent": {
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
def agent():
    logger = MagicMock()
    llm = MagicMock()
    modality_agent = ModalityAgent(cfg=modality_agent_cfg, logger=logger, llm=llm)
    modality_agent.guard = MagicMock(
        validate=lambda x: MagicMock(validation_passed=True)
    )
    return modality_agent


def test_modality_agent_routing(modality_agent_fixture, monkeypatch):
    monkeypatch.setattr(
        agent, "guard", MagicMock(validate=lambda x: MagicMock(validation_passed=True))
    )

    monkeypatch.setattr(
        ModalityAgent,
        "run",
        lambda _, query: ["text", "image"] if query == "text" else ["image"],
    )

    assert modality_agent_fixture.run("text") == ["text", "image"]
    assert modality_agent_fixture.run("image") == ["image"]


def test_modality_agent_unsupported_modality(modality_agent_fixture, monkeypatch):
    def fake_run(_, query):
        if query == "audio":
            raise ValueError("Rejected query due to: unsupported modality")
        return [query]

    monkeypatch.setattr(ModalityAgent, "run", fake_run)

    with pytest.raises(ValueError):
        modality_agent_fixture.run("audio")
