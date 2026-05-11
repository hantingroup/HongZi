from typing import Sequence

from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import fastapi
from fastapi import HTTPException, Request

from tianzi import tianzi_core, get_all_lex_names, get_lex_meta
from tianzi.phparser import helps, CALL_STACK
from tianzi.lexloaders import LexLoader


app = fastapi.FastAPI()


@app.get("/")
def translator_help():
    return helps()


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
        callstack=CALL_STACK
    )


@app.post("/translate", response_model=TranslateResponse)
async def translate_post(req: TranslateRequest):
    translated = await tianzi_core(req.text, req.variables)
    return TranslateResponse(
        translated=translated,
        callstack=CALL_STACK
    )


@app.get("/list")
async def all_lex_names():
    return await get_all_lex_names()


@app.get("/list/{name}")
async def lex_values(name: str) -> Sequence[str]:
    if lex := LexLoader.loaded.get(name):
        data = await lex.data()
        if isinstance(data, str):
            return list(data)
        return data.keys()
    raise HTTPException(status_code=404, detail=f"词库 {name} 不存在")


@app.get("/meta/{name}", response_class=PlainTextResponse)
async def meta(name: str):
    return await get_lex_meta(name)


@app.get("/find/{word}")
async def find(word: str) -> list[str]:
    lexs = await LexLoader.search_word(word)
    if not lexs:
        return []
    return [lex.meta.name for lex in lexs]
