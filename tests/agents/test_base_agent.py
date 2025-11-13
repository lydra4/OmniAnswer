from unittest.mock import MagicMock

from crewai.tools import BaseTool
from omegaconf import OmegaConf

from src.agents.base_agent.base_agent_task import BaseAgentTask


class AgentTaskImpl(BaseAgentTask):
    def _parse_result(self, result):
        return result


def test_create_task_renders_template_and_returns_task(
    logger, llm, ba_mod, dummy_output
):
    cfg = OmegaConf.create(
        {
            "agent": {"role": "tester", "name": "TestAgent"},
            "task": {"prompt": "Query: {query}", "meta": "static"},
        }
    )

    ba_mod.Agent = MagicMock(return_value="agent-object")
    ba_mod.Task = lambda **kwargs: kwargs

    output = dummy_output
    agent = AgentTaskImpl(cfg=cfg, logger=logger, llm=llm, output=output)

    task = agent.create_task("hello")
    assert agent is not None
    assert task["prompt"] == "Query: hello"
    assert task["meta"] == "static"
    assert task["agent"] == "agent-object"
    assert task["output_json"] is output


def test_create_agent_calls_agent_with_llm_and_tools_and_cfg_keys(
    logger, llm, ba_mod, dummy_tool, dummy_output
):
    cfg = OmegaConf.create({"agent": {"role": "r1", "extra": "v"}, "task": {}})

    ba_mod.Agent = MagicMock(return_value="agent-object")
    ba_mod.Task = MagicMock()

    output = dummy_output
    agent = AgentTaskImpl(
        cfg=cfg, logger=logger, llm=llm, output=output, tools=[dummy_tool]
    )

    ba_mod.Agent.assert_called_once()
    called_kwargs = ba_mod.Agent.call_args.kwargs
    tools_arg = called_kwargs.get("tools")
    assert agent is not None
    assert called_kwargs.get("llm") is llm
    assert isinstance(tools_arg, list)
    assert len(tools_arg) == 1
    assert isinstance(tools_arg[0], BaseTool)
    assert called_kwargs.get("role") == "r1"
    assert called_kwargs.get("extra") == "v"


def test_create_task_fills_query_in_multiple_fields(logger, llm, ba_mod, dummy_output):
    cfg = OmegaConf.create(
        {"agent": {"role": "r2"}, "task": {"p1": "A {query}", "p2": "B {query} end"}}
    )

    ba_mod.Agent = MagicMock(return_value="agent-object")
    ba_mod.Task = lambda **kwargs: kwargs

    output = dummy_output
    agent = AgentTaskImpl(cfg=cfg, logger=logger, llm=llm, output=output)

    task = agent.create_task("Q123")
    assert agent is not None
    assert task["p1"] == "A Q123"
    assert task["p2"] == "B Q123 end"


def test_logger_info_called_on_agent_initialization(logger, llm, ba_mod, dummy_output):
    cfg = OmegaConf.create({"agent": {"role": "logger-role"}, "task": {}})

    ba_mod.Agent = MagicMock(return_value="agent-object")
    ba_mod.Task = MagicMock()

    output = dummy_output
    AgentTaskImpl(cfg=cfg, logger=logger, llm=llm, output=output)

    logger.info.assert_called()
    logger.info.assert_called_with("'logger-role' successfully initialized.")
