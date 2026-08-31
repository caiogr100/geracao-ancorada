"""Registro de procedência: o que o repositório versiona no lugar dos PDF."""

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from .aquisicao import Registro


def escrever_procedencia(registros: Iterable[Registro], caminho: Path) -> None:
    fontes = sorted((asdict(r) for r in registros), key=lambda r: r["id"])
    caminho.write_text(
        json.dumps({"fontes": fontes}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
