"""BM25 com a estatística congelada sobre o índice inteiro.

Duas escolhas aqui têm consequência e ficam declaradas, porque as duas
mudam o resultado de um jeito que nenhum teste de fórmula pegaria.

A primeira é a forma da IDF. Sem lista de palavra funcional no índice, o
corpus tem termo em 81% dos pedaços, e a fórmula clássica de Robertson,
ln((N - n + 0,5) / (n + 0,5)), fica negativa acima de metade: com
N = 4.499 e n = 3.646, que é o "de" medido nos onze documentos, ela dá
-1,45. Escore negativo significa que o pedaço é punido por conter o termo
da pergunta. A forma do Lucene, ln(1 + (N - n + 0,5) / (n + 0,5)), tem piso
em zero e no mesmo caso dá +0,21.

A segunda é o filtro. Ele entra como máscara no laço de pontuação e nunca
como reíndice. Reindexar sobre o subconjunto mudaria N, o comprimento médio
e a frequência de documento ao mesmo tempo, e a ablação com-filtro contra
sem-filtro passaria a medir o escopo e uma mudança da função de pontuação
juntos, sem dizer qual das duas produziu a diferença.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

K1 = 1.2
"""Saturação da frequência do termo. Fixa, declarada, sem varredura."""

B = 0.75
"""Peso da normalização por comprimento do pedaço. Idem."""


@dataclass(frozen=True)
class BM25:
    """O índice léxico já construído, com a estatística congelada."""

    postings: dict[str, np.ndarray]
    """Para cada termo, a matriz de duas linhas [documento, frequência]."""

    comprimentos: np.ndarray
    n_documentos: int
    comprimento_medio: float
    k1: float = K1
    b: float = B
    _idf: dict[str, float] = field(default_factory=dict, repr=False)

    def df(self, token: str) -> int:
        """Em quantos pedaços o termo aparece."""
        entrada = self.postings.get(token)
        return 0 if entrada is None else int(entrada.shape[1])

    def idf(self, token: str) -> float:
        return self._idf.get(token, 0.0)

    def pontuar(
        self, tokens: list[str], mascara: np.ndarray | None = None
    ) -> np.ndarray:
        """O escore de cada pedaço do índice para esta consulta.

        Devolve sempre um vetor do tamanho do índice inteiro. Quem não passa
        no filtro sai com zero, e não fora do vetor, para que a posição
        continue sendo o identificador do pedaço em todo o caminho.
        """
        escores = np.zeros(self.n_documentos, dtype=np.float64)
        if self.n_documentos == 0:
            return escores

        denominador_base = self.k1 * (
            1 - self.b + self.b * self.comprimentos / self.comprimento_medio
        )

        for token in tokens:
            entrada = self.postings.get(token)
            if entrada is None:
                continue
            documentos, frequencias = entrada[0], entrada[1].astype(np.float64)
            saturada = frequencias * (self.k1 + 1) / (
                frequencias + denominador_base[documentos]
            )
            escores[documentos] += self._idf.get(token, 0.0) * saturada

        if mascara is not None:
            # Só bool: `~` sobre inteiro é complemento de bits, e `~[1, 0]`
            # dá `[-2, -1]`, que indexa de trás para a frente sem erro nenhum.
            mascara = np.asarray(mascara)
            if mascara.dtype != np.bool_ or mascara.shape != (self.n_documentos,):
                raise ValueError(
                    f"a máscara tem que ser bool com {self.n_documentos} posições, "
                    f"e veio {mascara.dtype} com forma {mascara.shape}"
                )
            escores[~mascara] = 0.0
        return escores


def construir(docs: list[list[str]], k1: float = K1, b: float = B) -> BM25:
    """Monta o índice léxico a partir dos tokens de cada pedaço."""
    n = len(docs)
    comprimentos = np.array([len(d) for d in docs], dtype=np.float64)
    medio = float(comprimentos.mean()) if n else 0.0

    bruto: dict[str, list[tuple[int, int]]] = {}
    for posicao, tokens in enumerate(docs):
        for token, frequencia in Counter(tokens).items():
            bruto.setdefault(token, []).append((posicao, frequencia))

    postings = {
        token: np.array(pares, dtype=np.int64).T for token, pares in bruto.items()
    }
    idf = {
        token: float(np.log(1 + (n - entrada.shape[1] + 0.5) / (entrada.shape[1] + 0.5)))
        for token, entrada in postings.items()
    }

    return BM25(
        postings=postings,
        comprimentos=comprimentos,
        n_documentos=n,
        comprimento_medio=medio or 1.0,
        k1=k1,
        b=b,
        _idf=idf,
    )
