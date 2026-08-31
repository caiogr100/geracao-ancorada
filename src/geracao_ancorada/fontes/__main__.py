"""Reconstrói o corpus de fontes a partir do manifesto.

    python -m geracao_ancorada.fontes
"""

import sys
from pathlib import Path

from .aquisicao import adquirir
from .http import baixar
from .manifesto import carregar_manifesto
from .procedencia import escrever_procedencia

RAIZ = Path(__file__).resolve().parents[3]
MANIFESTO = RAIZ / "fontes" / "manifesto.yaml"
CACHE = RAIZ / "fontes" / "cache"
PROCEDENCIA = RAIZ / "fontes" / "procedencia.json"


def main() -> int:
    registros = []
    falhas = 0
    for fonte in carregar_manifesto(MANIFESTO):
        try:
            registro = adquirir(fonte, destino=CACHE, baixar=baixar)
        except Exception as erro:
            print(f"FALHOU  {fonte.id}: {erro}")
            falhas += 1
            continue
        registros.append(registro)
        print(f"ok      {fonte.id}  {registro.sha256[:12]}  {registro.bytes} bytes")

    if registros:
        escrever_procedencia(registros, PROCEDENCIA)
        print(f"\nprocedência em {PROCEDENCIA.relative_to(RAIZ)}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
