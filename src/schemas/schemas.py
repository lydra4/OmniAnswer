"""Pydantic and TypedDict schemas used across the OmniAnswer pipeline."""

from typing import Dict, List, TypedDict

from pydantic import BaseModel


class StringOutput(BaseModel):
    """Pydantic schema representing a single URL string."""

    url: str


class ResultItem(TypedDict):
    """TypedDict describing a single modality-specific recommendation."""

    modality: str
    paraphrase: str
    url: str


class ResultDictFile(TypedDict):
    """TypedDict representing the full result set for a single query."""

    query: str
    results: List[ResultItem]


class DictOutput(BaseModel):
    """Generic key/value mapping used as agent JSON output."""

    items: Dict[str, str]


class StringListOutput(BaseModel):
    """Schema for a list of strings returned by an agent."""

    items: List[str]
