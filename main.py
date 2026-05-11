import fastapi

from fastapi.responses import PlainTextResponse

from tianzi.phparser import translate, helps, reset


app = fastapi.FastAPI()


@app.get("/", response_class=PlainTextResponse)
def read_root():
    return helps()


@app.get("/translate", response_class=PlainTextResponse)
async def translate_get(text: str):
    reset()
    return await translate(text)


@app.post("/translate", response_class=PlainTextResponse)
async def translate_post(request: fastapi.Request):
    body_bytes = await request.body()
    text = body_bytes.decode("utf-8")
    reset()
    return await translate(text)
