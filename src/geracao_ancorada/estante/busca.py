"""A busca híbrida sobre a estante carregada: dois braços, fusão e entrega.

O braço denso pontua pelo produto interno com a matriz normalizada; o braço
léxico pelo BM25 do índice. Cada braço devolve os seus primeiros, na
profundidade declarada, e a lista léxica para no último pedaço com escore
acima de zero: quem não casa token nenhum não está na lista e não ganha ponto
na fusão por constar dela. O filtro é uma máscara sobre o índice inteiro, e
não muda o escore de quem sobrevive a ele.

O resultado guarda os escores crus dos dois braços e a posição em cada lista,
porque o limiar de "não achei" e a ablação com reordenador precisam deles. O
reordenador é um gancho que hoje é a identidade.
"""

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from geracao_ancorada.estante import lexico, vetores
from geracao_ancorada.estante.fusao import (
    CONSTANTE_RRF,
    GARANTIDOS_POR_BRACO,
    N_ENTREGUE,
    PROFUNDIDADE,
    aplicar_garantia,
    rrf,
)
from geracao_ancorada.estante.indice import Estante, digerir
from geracao_ancorada.fatiamento.pedacos import Pedaco


@dataclass(frozen=True)
class Filtro:
    """O que o roteador decide; a busca só aplica.

    As áreas casam por interseção, e não por igualdade: três dos onze
    documentos têm duas áreas, e o calendário vacinal sustenta sozinho
    pediatria e preventiva.
    """

    areas: tuple[str, ...] = ()
    anos: tuple[int, ...] = ()
    fontes: tuple[str, ...] = ()
    tipos: tuple[str, ...] = ()
    incluir_descartaveis: bool = False

    def aceita(self, p: Pedaco) -> bool:
        if p.descartavel and not self.incluir_descartaveis:
            return False
        if self.areas and not set(self.areas) & set(p.areas):
            return False
        if self.anos and p.ano not in self.anos:
            return False
        if self.fontes and p.fonte_id not in self.fontes:
            return False
        if self.tipos and p.tipo not in self.tipos:
            return False
        return True

    def mascara(self, pedacos: list[Pedaco]) -> np.ndarray:
        return np.fromiter((self.aceita(p) for p in pedacos), dtype=np.bool_, count=len(pedacos))


@dataclass(frozen=True)
class Achado:
    """Um pedaço candidato, com a procedência do escore em cada braço."""

    id: str
    sha256_texto: str
    pedaco: Pedaco
    posicao_densa: int | None
    posicao_lexica: int | None
    cosseno: float
    bm25: float
    rrf: float
    garantido: bool


@dataclass(frozen=True)
class Busca:
    """Os `n` entregues, iteráveis, e todos os candidatos que os dois braços trouxeram."""

    consulta: str
    entrega: list[Achado]
    candidatos: list[Achado]
    n: int
    profundidade: int

    def __iter__(self) -> Iterator[Achado]:
        return iter(self.entrega)

    def __len__(self) -> int:
        return len(self.entrega)

    def __getitem__(self, i):
        return self.entrega[i]


def _ordenar(escores: np.ndarray, ids: list[str], mascara: np.ndarray) -> list[int]:
    """As posições do índice, do maior escore ao menor, empate pelo id."""
    vivos = [i for i in range(len(ids)) if mascara[i]]
    return sorted(vivos, key=lambda i: (-float(escores[i]), ids[i]))


def buscar(
    estante: Estante,
    consulta: str,
    *,
    vetor: np.ndarray | None = None,
    filtro: Filtro | None = None,
    n: int = N_ENTREGUE,
    profundidade: int = PROFUNDIDADE,
    reordenador: Callable[[str, list[Achado]], list[Achado]] | None = None,
    vetorizar: Callable[[str], np.ndarray] = vetores.vetorizar_consulta,
) -> Busca:
    """Os dois braços, fundidos, com o primeiro de cada um garantido.

    Quem já tem o vetor da consulta passa `vetor` e o servidor de embedding
    não é chamado; na geração, o modelo de embedding não cabe na placa ao
    lado do gerador.
    """
    filtro = filtro or Filtro()
    mascara = filtro.mascara(estante.pedacos)
    ids = [p.id for p in estante.pedacos]

    if vetor is None:
        vetor = vetorizar(consulta)
    cossenos = estante.matriz @ np.asarray(vetor, dtype=estante.matriz.dtype)
    escores_bm25 = estante.bm25.pontuar(lexico.tokenizar_consulta(consulta), mascara)

    densa = _ordenar(cossenos, ids, mascara)[:profundidade]
    lexica = [i for i in _ordenar(escores_bm25, ids, mascara) if escores_bm25[i] > 0][:profundidade]

    listas = {"lexico": [ids[i] for i in lexica], "denso": [ids[i] for i in densa]}
    fundida = rrf(listas)
    entrega = aplicar_garantia(listas, fundida, n=n)

    posicao_densa = {ids[i]: pos for pos, i in enumerate(densa, start=1)}
    posicao_lexica = {ids[i]: pos for pos, i in enumerate(lexica, start=1)}
    indice_por_id = {id_: i for i, id_ in enumerate(ids)}
    entregues = {id_ for id_, _, _ in entrega}
    marcado = {id_: g for id_, _, g in entrega}

    def achado(id_: str, escore: float) -> Achado:
        i = indice_por_id[id_]
        return Achado(
            id=id_,
            sha256_texto=digerir(estante.pedacos[i].texto),
            pedaco=estante.pedacos[i],
            posicao_densa=posicao_densa.get(id_),
            posicao_lexica=posicao_lexica.get(id_),
            cosseno=float(cossenos[i]),
            bm25=float(escores_bm25[i]),
            rrf=escore,
            garantido=marcado.get(id_, False),
        )

    candidatos = [achado(id_, escore) for id_, escore, _ in entrega]
    candidatos += [achado(id_, escore) for id_, escore in fundida if id_ not in entregues]
    if reordenador is not None:
        candidatos = reordenador(consulta, candidatos)
    return Busca(consulta=consulta, entrega=candidatos[:n], candidatos=candidatos, n=n,
                 profundidade=profundidade)


def registrar(consulta: str, busca: Busca, caminho: Path) -> None:
    """Acrescenta a corrida ao arquivo JSONL: todos os candidatos, sem o texto.

    O que fica gravado é o que permite recalcular a fusão sem garantia, ou com
    outra constante, sem rodar a busca de novo.
    """
    corrida = {
        "consulta": consulta,
        "n": busca.n,
        "profundidade": busca.profundidade,
        "constante_rrf": CONSTANTE_RRF,
        "garantidos_por_braco": GARANTIDOS_POR_BRACO,
        "candidatos": [
            {k: v for k, v in asdict(a).items() if k != "pedaco"} for a in busca.candidatos
        ],
    }
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(corrida, ensure_ascii=False) + "\n")
