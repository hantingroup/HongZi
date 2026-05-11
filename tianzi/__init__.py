import logging
import traceback
from typing import cast

from fuzzywuzzy import process
import regex

from . import lexloaders
from .phparser import BreakOut, reset, translate

logger = logging.getLogger("tianzi")


CRASHACTER = regex.compile(
    "[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f\u1fff\u200b-\u200f\u2028-\u202e\u2060-\u206f\uf900-\ufaff\U000107c0-\U000107ff\U000108b0-\U000108df]"
)


async def tianzi_core(raw: str, initial_var: dict | None = None) -> str:
    try:
        reset(initial_var)
        translated: str = str(await translate(raw, final=True))
    except BreakOut as bo:
        translated = bo.args[0]
    except TimeoutError:
        translated = "正则超时。您的输入可能太长或太复杂。"
    except Exception as e:
        translated = f"意外的错误：{repr(e)}，栈如下：\n{traceback.format_exc()}\n请联系找北。"
    if phparser.CurrentStat.censor:
        translated = translated  # censor(translated)
    ret_msg = regex.sub(
        CRASHACTER, lambda char: f"U+{hex(ord(char.group(0)))[2:].upper()}", translated.removeprefix("\n").removeprefix(" "))
    logger.info(f"CALL STACK: {"".join(phparser.CALL_STACK)}")

    return ret_msg


async def tianzi_callstack():
    return "".join(phparser.CALL_STACK)


async def get_all_lex_names():
    return list(lexloaders.LexLoader.loaded.keys())


async def get_lex_meta(name: str):
    try:
        meta = await lexloaders.LexLoader.loaded[name].show_meta()
    except KeyError:
        fuzz = cast(list[tuple[str, int]], process.extract(
            name, list(lexloaders.LexLoader.loaded.keys()), limit=2))
        if fuzz and fuzz[0][1] >= 70:
            if len(fuzz) == 2 and fuzz[1][1] == fuzz[0][1]:
                hint = f"猜你想找：「{fuzz[0][0]}」「{fuzz[1][0]}」"
            else:
                hint = f"猜你想找：「{fuzz[0][0]}」"
        else:
            hint = ""
        meta = f"词库不存在。{hint}"
    return meta


async def find_lex(word: str):
    lexs = await lexloaders.LexLoader.search_word(word)
    if not lexs:
        ret = f"{word} - 不见于任何词典"
    else:
        lex_names = [lex.meta.name for lex in lexs]
        ret = f"{word} - 见于以下{len(lexs)}个词典：「{"」「".join(lex_names)}」"
    return ret
