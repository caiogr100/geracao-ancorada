"""Constrói o índice a partir do manifesto e mostra a assinatura na tela.

    python -m geracao_ancorada.estante indexar
    python -m geracao_ancorada.estante conferir

`indexar` reconstrói tudo: fatia os PDF que estiverem baixados, vetoriza pelo
Ollama o que o cache ainda não tem e grava os dois destinos. `conferir` carrega
o que está em disco e se recusa se matriz, registro e assinatura divergirem ou
se o modelo no ar não for o que construiu o índice.
"""

import argparse
import sys

from ..fontes.manifesto import carregar_manifesto
from .indice import (
    IndiceDeOutroModelo,
    IndiceDesalinhado,
    RAIZ,
    REGISTRO,
    carregar,
    indexar,
)

MANIFESTO = RAIZ / "fontes" / "manifesto.yaml"


def _mostrar(assinatura) -> None:
    print(f"pedaços      {assinatura.n_pedacos}")
    print(f"fontes       {', '.join(assinatura.fontes_indexadas)}")
    for fonte_id, motivo in assinatura.fontes_de_fora.items():
        print(f"de fora      {fonte_id}  ({motivo})")
    print(f"recusados    {len(assinatura.recusados)}")
    for r in assinatura.recusados:
        print(f"             {r['id']}  {r['n_caracteres']} caracteres")
    print(f"modelo       {assinatura.modelo}  {assinatura.digest_modelo[:12]}  ollama {assinatura.versao_ollama}")
    print(f"léxico       {assinatura.bm25['vocabulario']} termos, comprimento médio {assinatura.bm25['comprimento_medio']:.1f}")
    print(f"fatiador     {assinatura.commit_fatiador}  {assinatura.data}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="geracao_ancorada.estante")
    ap.add_argument("comando", choices=["indexar", "conferir"])
    ap.add_argument("--incluir-superadas", action="store_true",
                    help="índice próprio com a fonte superada, para a sonda contrafactual")
    args = ap.parse_args(argv)

    if args.comando == "indexar":
        assinatura = indexar(carregar_manifesto(MANIFESTO), incluir_superadas=args.incluir_superadas)
        _mostrar(assinatura)
        print(f"\nassinatura em {(REGISTRO / 'assinatura.json').relative_to(RAIZ)}")
        return 0

    try:
        estante = carregar()
    except (IndiceDesalinhado, IndiceDeOutroModelo) as erro:
        print(f"RECUSADO  {erro}", file=sys.stderr)
        return 1
    _mostrar(estante.assinatura)
    print(f"\nmatriz       {estante.matriz.shape[0]} x {estante.matriz.shape[1]}, alinhada com o registro")
    return 0


if __name__ == "__main__":
    sys.exit(main())
