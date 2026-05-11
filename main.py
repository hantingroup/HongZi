import fastapi

from fastapi.responses import PlainTextResponse

from tianzi import tianzi_core, tianzi_callstack, get_all_lex_names, get_lex_meta, find_lex
from tianzi.phparser import helps


app = fastapi.FastAPI()


@app.get("/", response_class=PlainTextResponse)
def read_root():
    return helps()


@app.get("/translate", response_class=PlainTextResponse)
@app.post("/translate", response_class=PlainTextResponse)
async def translate(text: str, variables: dict[str, str] | None = None):
    return await tianzi_core(text, variables)

@app.get("/callstack", response_class=PlainTextResponse)
async def callstack():
    return await tianzi_callstack()

@app.get("/list")
async def all_lex_names():
    return await get_all_lex_names()

@app.get("/meta", response_class=PlainTextResponse)
async def meta(name: str):
    return await get_lex_meta(name)

@app.get("/find", response_class=PlainTextResponse)
async def find(name: str):
    return await find_lex(name)
