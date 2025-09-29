from pydantic import BaseModel


class TextSchema(BaseModel):
    text: str
