from typing import Dict, List

from pydantic import BaseModel


class StringListOutput(BaseModel):
    items: List[str]


class DictOutput(BaseModel):
    items: Dict[str, str]


class StringOutput(BaseModel):
    items: str
