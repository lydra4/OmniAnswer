import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel


def install_fake_crewai():
    if "crewai" in sys.modules:
        return

    fake = types.ModuleType("crewai")

    class FakeLLM:
        pass

    def fake_agent_factory(*args, **kwargs):
        return SimpleNamespace(args=args, kwargs=kwargs)

    def fake_task_factory(**kwargs):
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

    def _run(self, *args, **kwargs):
        return "ok"


@pytest.fixture
def logger():
    return MagicMock()


@pytest.fixture
def llm():
    return MagicMock()


@pytest.fixture
def dummy_output():
    return DummyOutput()


@pytest.fixture
def dummy_tool():
    return DummyTool()


@pytest.fixture
def ba_mod():
    import src.agents.base_agent.base_agent_task as ba_mod

    return ba_mod
