from typing import List, TypedDict

from pydantic import BaseModel


class StringOutput(BaseModel):
    url: str


class ResultItem(TypedDict):
    modality: str
    paraphrase: str
    url: str


class ResultDictFile(TypedDict):
    query: str
    results: List[ResultItem]
