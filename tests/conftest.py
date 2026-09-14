"""Fixtures compartilhadas, incluindo o corpus de verdade.

O corpus é caro: extrair os onze PDF leva minutos. Por isso ele é fixture de
sessão e só é montado quando algum teste marcado `corpus` pede. O marcador
fica fora da execução padrão, e a varredura roda com `-m corpus`.
"""

from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
MANIFESTO = RAIZ / "fontes" / "manifesto.yaml"
CACHE = RAIZ / "fontes" / "cache"


@pytest.fixture(scope="session")
def pedacos_do_corpus():
    """Todos os pedaços úteis dos documentos que estiverem baixados."""
    from geracao_ancorada.fatiamento.extracao import extrair
    from geracao_ancorada.fatiamento.pedacos import fatiar
    from geracao_ancorada.fontes.manifesto import carregar_manifesto

    if not CACHE.exists():
        pytest.skip("corpus não baixado; rode `python -m geracao_ancorada.fontes`")

    fontes = {f.id: f for f in carregar_manifesto(MANIFESTO)}
    pedacos = []
    for caminho in sorted(CACHE.glob("*.pdf")):
        fonte = fontes.get(caminho.stem)
        if fonte is None:
            continue
        pedacos.extend(p for p in fatiar(fonte, extrair(caminho)) if not p.descartavel)

    if not pedacos:
        pytest.skip("nenhum documento do manifesto está baixado")
    return pedacos
