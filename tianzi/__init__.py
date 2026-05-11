import logging
import traceback
import regex

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
    ret_msg = regex.sub(
        CRASHACTER, lambda char: f"U+{hex(ord(char.group(0)))[2:].upper()}", translated.removeprefix("\n").removeprefix(" "))
    logger.info(f"CALL STACK: {"".join(phparser.CALL_STACK)}")

    return ret_msg
