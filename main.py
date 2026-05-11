from typing import Sequence

from fuzzywuzzy import process
from pydantic import BaseModel
import fastapi
from fastapi import HTTPException, Request

from tianzi import tianzi_core, phparser
from tianzi.lexloaders import LexLoader


app = fastapi.FastAPI()


@app.get("/")
def translator_help():
    return {
        translator.__name__: translator.__doc__ or ""
        for (_, translator) in phparser.translators
    }


class TranslateRequest(BaseModel):
    text: str
    variables: dict[str, str] | None = None


class TranslateResponse(BaseModel):
    translated: str
    callstack: list[str]


@app.get("/translate/{text}", response_model=TranslateResponse)
async def translate_get(text: str, request: Request):
    translated = await tianzi_core(text, dict(request.query_params))
    return TranslateResponse(
        translated=translated,
        callstack=phparser.CALL_STACK
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate_post(req: TranslateRequest):
    translated = await tianzi_core(req.text, req.variables)
    return TranslateResponse(
        translated=translated,
        callstack=phparser.CALL_STACK
    )


@app.get("/list")
def all_lex_names():
    return list(LexLoader.loaded.keys())


@app.get("/list/{name}")
async def lex_values(name: str) -> Sequence[str]:
    if lex := LexLoader.loaded.get(name):
        data = await lex.data()
        if isinstance(data, str):
            return list(data)
        return data.keys()
    raise HTTPException(status_code=404, detail=f"{name} not found")


@app.get("/meta/{name}")
async def meta(name: str):
    if lex := LexLoader.loaded.get(name):
        return lex.meta
    raise HTTPException(status_code=404, detail=f"{name} not found")


@app.get("/find/{word}")
async def find(word: str) -> list[str]:
    lexs = await LexLoader.search_word(word)
    if not lexs:
        return []
    return [lex.meta.name for lex in lexs]


@app.get("/fuzzy/{name}")
async def fuzzy(name: str, limit: int = 5):
    return process.extract(name, list(LexLoader.loaded.keys()), limit=limit)
