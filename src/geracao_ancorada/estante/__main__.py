"""Constrói o índice, confere o que está em disco e busca nele.

    python -m geracao_ancorada.estante indexar
    python -m geracao_ancorada.estante conferir
    python -m geracao_ancorada.estante buscar "pergunta" [--n 8] [--area pediatria] [--json] [--saida corridas.jsonl]

`indexar` reconstrói tudo: fatia os PDF que estiverem baixados, vetoriza pelo
Ollama o que o cache ainda não tem e grava os dois destinos. `conferir` carrega
o que está em disco e se recusa se matriz, registro e assinatura divergirem ou
se o modelo no ar não for o que construiu o índice.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ..fontes.manifesto import carregar_manifesto
from .busca import Filtro, buscar, registrar
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


def _mostrar_busca(busca) -> None:
    for posicao, a in enumerate(busca, start=1):
        p = a.pedaco
        braco = (
            f"L{a.posicao_lexica or '-':<3}D{a.posicao_densa or '-':<3}"
        )
        marca = "garantido" if a.garantido else ""
        print(f"{posicao:>2}. {a.id}")
        print(f"    {braco} cosseno {a.cosseno:.3f}  bm25 {a.bm25:6.2f}  rrf {a.rrf:.4f}  {marca}")
        print(f"    {p.cabecalho}")
        print(f"    {' '.join(p.texto.split())[:160]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="geracao_ancorada.estante")
    ap.add_argument("comando", choices=["indexar", "conferir", "buscar"])
    ap.add_argument("consulta", nargs="?", help="a pergunta, para `buscar`")
    ap.add_argument("--incluir-superadas", action="store_true",
                    help="índice próprio com a fonte superada, para a sonda contrafactual")
    ap.add_argument("--n", type=int, default=8, help="quantos pedaços entregar")
    ap.add_argument("--area", action="append", default=[], help="área do roteador; repetível")
    ap.add_argument("--fonte", action="append", default=[], help="id de fonte; repetível")
    ap.add_argument("--json", action="store_true", help="saída em JSON, sem o texto")
    ap.add_argument("--saida", type=Path, help="arquivo JSONL onde a corrida é acrescentada")
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
    if args.comando == "conferir":
        _mostrar(estante.assinatura)
        print(f"\nmatriz       {estante.matriz.shape[0]} x {estante.matriz.shape[1]}, alinhada com o registro")
        return 0

    if not args.consulta:
        ap.error("`buscar` precisa da consulta")
    filtro = Filtro(areas=tuple(args.area), fontes=tuple(args.fonte))
    busca = buscar(estante, args.consulta, filtro=filtro, n=args.n)
    if args.saida:
        registrar(args.consulta, busca, args.saida)
    if args.json:
        linhas = [{k: v for k, v in asdict(a).items() if k != "pedaco"} for a in busca]
        print(json.dumps(linhas, ensure_ascii=False, indent=1))
    else:
        _mostrar_busca(busca)
    return 0


if __name__ == "__main__":
    sys.exit(main())
