import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel


def install_fake_crewai() -> None:
    if "crewai" in sys.modules:
        return

    fake = types.ModuleType("crewai")

    class FakeLLM:
        pass

    def fake_agent_factory(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(args=args, kwargs=kwargs)

    def fake_task_factory(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            **kwargs, execute_sync=lambda: SimpleNamespace(json='{"items": []}')
        )

    fake.LLM = object
    fake.Agent = fake_agent_factory
    fake.Task = fake_task_factory

    sys.modules["crewai"] = fake

    tools_mod = types.ModuleType("crewai.tools")
    tools_mod.BaseTool = object
    sys.modules["crewai.tools"] = tools_mod

    tasks_mod = types.ModuleType("crewai.tasks")
    sys.modules["crewai.tasks"] = tasks_mod

    task_output_mod = types.ModuleType("crewai.tasks.task_output")

    class TaskOutput:
        def __init__(self, json=None) -> None:
            self.json = json

    task_output_mod.TaskOutput = TaskOutput
    sys.modules["crewai.tasks.task_output"] = task_output_mod


install_fake_crewai()


class DummyOutput(BaseModel):
    result: str = ""


class DummyTool:
    """Simple tool class used in tests. The real BaseTool isn't required here; the
    test asserts that tools forwarded are list-like and instances of something
    subclassing the placeholder BaseTool. The fake `crewai.tools.BaseTool` is
    provided by install_fake_crewai earlier, and DummyTool is compatible for
    testing purposes."""

    name: str = "Dummy Tool"
    description: str = "Dummy Tool"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return "ok"


@pytest.fixture
def logger() -> MagicMock:
    return MagicMock()


@pytest.fixture
def llm() -> MagicMock:
    return MagicMock()


@pytest.fixture
def dummy_output() -> DummyOutput:
    return DummyOutput()


@pytest.fixture
def dummy_tool() -> DummyTool:
    return DummyTool()


@pytest.fixture
def ba_mod() -> Any:
    import src.agents.base_agent.base_agent_task as ba_mod

    return ba_mod
