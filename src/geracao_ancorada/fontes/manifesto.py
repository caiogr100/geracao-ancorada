"""Leitura do manifesto de fontes normativas."""

from dataclasses import dataclass
from pathlib import Path

import yaml

CAMPOS = ("id", "titulo", "orgao", "ano", "url", "base_legal", "redistribuivel", "areas")


class ManifestoInvalido(Exception):
    """O manifesto não descreve a procedência exigida pelo plano de análise."""


@dataclass(frozen=True)
class Fonte:
    id: str
    titulo: str
    orgao: str
    ano: int
    url: str
    base_legal: str
    redistribuivel: bool
    areas: list[str]
    sha256: str | None = None
    substitui: str | None = None


def carregar_manifesto(caminho: Path) -> list[Fonte]:
    dados = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    fontes = []
    for entrada in dados["fontes"]:
        faltando = [campo for campo in CAMPOS if campo not in entrada]
        if faltando:
            identificacao = entrada.get("id", "<fonte sem id>")
            raise ManifestoInvalido(
                f"fonte {identificacao}: faltam os campos {', '.join(faltando)}"
            )
        if any(fonte.id == entrada["id"] for fonte in fontes):
            raise ManifestoInvalido(f"id repetido no manifesto: {entrada['id']}")
        fontes.append(Fonte(**entrada))

    ids = {fonte.id for fonte in fontes}
    for fonte in fontes:
        if fonte.substitui is not None and fonte.substitui not in ids:
            raise ManifestoInvalido(
                f"fonte {fonte.id}: substitui {fonte.substitui}, "
                "que não está no manifesto"
            )
    return fontes
