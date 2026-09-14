"""Fatia uma fonte do manifesto e mostra os pedaços na tela.

    python -m geracao_ancorada.fatiamento pcdt-dm2-2026 --amostra 20

O objetivo desta saída é o olho humano: cada pedaço aparece com a página e a
seção pra que dê pra responder, lendo só ele, de que doença fala, pra quem e de
que ano é. Se não der, o fatiador volta pra bancada antes de virar vetor.
"""

import argparse
import sys
from pathlib import Path

from ..fontes.manifesto import carregar_manifesto
from .extracao import extrair
from .pedacos import fatiar

RAIZ = Path(__file__).resolve().parents[3]
MANIFESTO = RAIZ / "fontes" / "manifesto.yaml"
CACHE = RAIZ / "fontes" / "cache"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="geracao_ancorada.fatiamento")
    ap.add_argument("fonte_id")
    ap.add_argument("--amostra", type=int, default=20, help="quantos pedaços mostrar")
    ap.add_argument("--tudo", action="store_true", help="mostra todos")
    ap.add_argument("--secao", help="só os pedaços desta seção (ex.: 8.3.4)")
    args = ap.parse_args(argv)

    fontes = {f.id: f for f in carregar_manifesto(MANIFESTO)}
    fonte = fontes.get(args.fonte_id)
    if fonte is None:
        print(f"fonte desconhecida: {args.fonte_id}", file=sys.stderr)
        print(f"disponíveis: {', '.join(sorted(fontes))}", file=sys.stderr)
        return 2

    caminho = CACHE / f"{fonte.id}.pdf"
    if not caminho.exists():
        print(f"não baixado: {caminho}. rode `python -m geracao_ancorada.fontes`", file=sys.stderr)
        return 2

    pedacos = fatiar(fonte, extrair(caminho))
    uteis = [p for p in pedacos if not p.descartavel]
    tabelas = [p for p in uteis if p.tipo == "tabela"]

    print(f"fonte      {fonte.id}")
    print(f"pedaços    {len(pedacos)}  ({len(uteis)} úteis, {len(pedacos) - len(uteis)} descartáveis)")
    print(f"tabelas    {len(tabelas)}")
    if uteis:
        tamanhos = sorted(len(p.texto) for p in uteis)
        print(f"tamanho    mín {tamanhos[0]}  mediana {tamanhos[len(tamanhos) // 2]}  máx {tamanhos[-1]} caracteres")

    mostrar = [p for p in uteis if args.secao is None or p.secao.startswith(args.secao)]
    if not args.tudo:
        passo = max(1, len(mostrar) // args.amostra) if args.amostra else 1
        mostrar = mostrar[::passo][: args.amostra] if args.amostra else []

    for pedaco in mostrar:
        print(f"\n{'-' * 78}\n[{pedaco.id}] {pedaco.tipo}\n{pedaco.cabecalho}\n")
        print(pedaco.texto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
