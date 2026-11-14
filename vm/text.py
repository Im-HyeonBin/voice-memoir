"""화면 문구에 쓰는 작은 한국어 도우미."""
from __future__ import annotations


def josa(word: str, with_final: str, without_final: str) -> str:
    """받침이 있으면 앞의 조사, 없으면 뒤의 조사를 붙인다(예: 형과, 아버지와).

    한글 음절이 아닌 글자(영문·숫자)로 끝나면 받침이 없는 것으로 본다.
    """
    ch = word[-1] if word else ""
    has_final = "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 != 0
    return word + (with_final if has_final else without_final)
