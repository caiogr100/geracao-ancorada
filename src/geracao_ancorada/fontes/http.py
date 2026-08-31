"""Baixador HTTP.

Cabeçalho de navegador porque parte dos portais de sociedade médica devolve
403 para requisição automatizada.
"""

import httpx

AGENTE = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


def baixar(url: str) -> bytes:
    resposta = httpx.get(
        url,
        headers={"User-Agent": AGENTE},
        follow_redirects=True,
        timeout=120.0,
    )
    resposta.raise_for_status()
    return resposta.content
