"""Aquisição das fontes do manifesto, com registro de procedência.

O repositório não guarda os PDF. Guarda o ponteiro, o hash e a data, e este
módulo é o que reconstrói o corpus a partir das fontes oficiais.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .manifesto import Fonte


class ConteudoInesperado(Exception):
    """A URL não devolveu o documento, e sim outra coisa (em geral a página)."""


class FonteDivergente(Exception):
    """O documento na URL oficial não é o que o snapshot congelou."""


@dataclass(frozen=True)
class Registro:
    id: str
    url: str
    sha256: str
    bytes: int
    baixado_em: str


def adquirir(fonte: Fonte, *, destino: Path, baixar: Callable[[str], bytes]) -> Registro:
    conteudo = baixar(fonte.url)

    if not conteudo.startswith(b"%PDF-"):
        raise ConteudoInesperado(
            f"fonte {fonte.id}: {fonte.url} não devolveu um PDF "
            f"(começa com {conteudo[:20]!r})"
        )

    obtido = hashlib.sha256(conteudo).hexdigest()

    if fonte.sha256 is not None and obtido != fonte.sha256:
        raise FonteDivergente(
            f"fonte {fonte.id}: o manifesto congelou {fonte.sha256}, "
            f"a URL devolveu {obtido}"
        )

    destino.mkdir(parents=True, exist_ok=True)
    (destino / f"{fonte.id}.pdf").write_bytes(conteudo)

    return Registro(
        id=fonte.id,
        url=fonte.url,
        sha256=obtido,
        bytes=len(conteudo),
        baixado_em=date.today().isoformat(),
    )
