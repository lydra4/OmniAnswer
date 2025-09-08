from typing import List

from pydantic import BaseModel


class StringListOutput(BaseModel):
    items: List[str]
