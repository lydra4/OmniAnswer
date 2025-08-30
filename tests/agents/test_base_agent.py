from unittest.mock import MagicMock

from omegaconf import OmegaConf

from src.agents.base_agent import BaseAgent


class WorkingAgent(BaseAgent):
    def run(self, query: str, **kwargs):
        return f"Processed: {query}"


def test_working_agent_run_returns_expected_output():
    cfg = OmegaConf.create(
        {
            "name": "WorkingAgent",
            "role": "test role",
            "description": "test description",
            "system_message": "test system message",
        }
    )
    logger = MagicMock()
    llm = MagicMock()

    agent = WorkingAgent(cfg=cfg, logger=logger, llm=llm)
    result = agent.run("hello")
    assert result == "Processed: hello"
