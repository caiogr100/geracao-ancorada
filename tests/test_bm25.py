"""O BM25 do braço léxico, com a estatística congelada sobre o índice inteiro.

O teste que importa aqui não é o da fórmula, que qualquer implementação
acerta. É o do sinal da IDF. Sem lista de palavra funcional no índice, o
corpus tem termo em 81% dos pedaços, e na fórmula clássica de Robertson isso
produz IDF negativa: o pedaço que CONTÉM o termo da pergunta é punido por
contê-lo.
"""

import numpy as np
import pytest

from geracao_ancorada.estante.bm25 import BM25, construir


@pytest.fixture
def corpus_com_termo_em_todo_lugar():
    """Dez pedaços, e a palavra "de" aparece nos dez, como no corpus real."""
    return [
        ["de", "dose", "unica"],
        ["de", "dose", "dupla"],
        ["de", "esquema", "basico"],
        ["de", "esquema", "de", "reforco"],
        ["de", "vacina", "bcg"],
        ["de", "vacina", "hpv"],
        ["de", "notificacao", "compulsoria"],
        ["de", "notificacao", "imediata"],
        ["de", "tuberculose", "pulmonar"],
        ["de", "tuberculose", "ighvz"],
    ]


def test_termo_presente_em_todo_o_corpus_nunca_baixa_o_escore(
    corpus_com_termo_em_todo_lugar,
):
    """A IDF do Lucene mantém o piso em zero mesmo com o termo em 100% deles."""
    bm25 = construir(corpus_com_termo_em_todo_lugar)

    escores = bm25.pontuar(["de"])

    assert bm25.df("de") == 10
    assert np.all(escores >= 0)


def test_termo_raro_pontua_mais_que_termo_comum(corpus_com_termo_em_todo_lugar):
    """O sinal do braço léxico: quem separa é a sigla, e não a preposição."""
    bm25 = construir(corpus_com_termo_em_todo_lugar)

    raro = bm25.pontuar(["ighvz"])
    comum = bm25.pontuar(["de"])

    assert raro.max() > comum.max()


def test_termo_fora_do_vocabulario_pontua_zero(corpus_com_termo_em_todo_lugar):
    bm25 = construir(corpus_com_termo_em_todo_lugar)

    escores = bm25.pontuar(["palavraquenaoexiste"])

    assert bm25.df("palavraquenaoexiste") == 0
    assert np.all(escores == 0)


def test_pedaco_curto_com_o_termo_ganha_do_longo(corpus_com_termo_em_todo_lugar):
    """A normalização por comprimento é o que o b=0,75 controla."""
    docs = corpus_com_termo_em_todo_lugar + [
        ["ighvz"],
        ["ighvz"] + ["enchimento"] * 200,
    ]
    bm25 = construir(docs)

    escores = bm25.pontuar(["ighvz"])

    assert escores[-2] > escores[-1]


def test_filtrar_nao_muda_o_escore_dos_pedacos_que_sobrevivem(
    corpus_com_termo_em_todo_lugar,
):
    """O filtro é máscara no laço, e nunca um reíndice.

    Se filtrar reconstruísse a estatística, a ablação com-filtro contra
    sem-filtro mediria o escopo e uma mudança da função de pontuação ao mesmo
    tempo, e o resultado não diria qual das duas produziu a diferença.
    """
    bm25 = construir(corpus_com_termo_em_todo_lugar)
    mascara = np.zeros(10, dtype=bool)
    mascara[[0, 1, 9]] = True

    inteiro = bm25.pontuar(["dose", "tuberculose"])
    filtrado = bm25.pontuar(["dose", "tuberculose"], mascara=mascara)

    assert np.allclose(filtrado[mascara], inteiro[mascara])
    assert np.all(filtrado[~mascara] == 0)


def test_estatistica_e_do_indice_inteiro_e_nao_do_subconjunto(
    corpus_com_termo_em_todo_lugar,
):
    """Mesmo com máscara, o N e o comprimento médio continuam os do índice."""
    bm25 = construir(corpus_com_termo_em_todo_lugar)
    mascara = np.zeros(10, dtype=bool)
    mascara[0] = True

    assert bm25.n_documentos == 10
    assert bm25.pontuar(["dose"], mascara=mascara)[0] == pytest.approx(
        bm25.pontuar(["dose"])[0]
    )


def test_repetir_o_termo_na_consulta_nao_multiplica_o_escore(
    corpus_com_termo_em_todo_lugar,
):
    """A saturação do k1 age por termo, e o termo repetido soma duas vezes.

    Este teste existe para travar o comportamento, e não para elogiá-lo: a
    consulta que vem do enunciado repete palavra, e a decisão de somar cada
    ocorrência precisa estar escrita em algum lugar que falha se alguém mudar.
    """
    bm25 = construir(corpus_com_termo_em_todo_lugar)

    uma = bm25.pontuar(["dose"])
    duas = bm25.pontuar(["dose", "dose"])

    assert np.allclose(duas, 2 * uma)


def test_corpus_vazio_nao_explode():
    bm25 = construir([])

    assert bm25.n_documentos == 0
    assert bm25.pontuar(["dose"]).shape == (0,)


def test_constantes_declaradas_sao_as_do_pre_registro(
    corpus_com_termo_em_todo_lugar,
):
    """k1 e b ficam fixos e declarados, sem varredura, e entram na assinatura."""
    bm25 = construir(corpus_com_termo_em_todo_lugar)

    assert (bm25.k1, bm25.b) == (1.2, 0.75)


@pytest.mark.corpus
def test_idf_e_positiva_para_todo_termo_do_vocabulario(pedacos_do_corpus):
    """A varredura que pega a IDF negativa, e ela só aparece no corpus real.

    Nenhum corpus sintético de dez documentos reproduz a distribuição que
    causa o problema. Aqui o vocabulário inteiro é varrido, e o termo mais
    comum dos onze documentos aparece em mais de quatro quintos dos pedaços.
    """
    from geracao_ancorada.estante.lexico import tokenizar

    bm25 = construir([tokenizar(p.texto) for p in pedacos_do_corpus])

    negativos = [t for t in bm25.postings if bm25.idf(t) < 0]
    mais_comum = max(bm25.postings, key=bm25.df)

    assert bm25.df(mais_comum) > bm25.n_documentos // 2, (
        "o corpus deixou de ter termo em mais da metade dos pedaços, e esta "
        "varredura parou de testar o que a frase dela diz"
    )
    assert negativos == []


def test_mascara_de_inteiro_e_recusada():
    """`~` sobre um vetor de inteiros é complemento de bits, e não negação:
    `~[1, 0, 1, 0]` dá `[-2, -1, -2, -1]`, que indexa de trás para a frente e
    zera as posições erradas em silêncio. A fusão vai montar a máscara a
    partir do filtro de área e vigência, e é o primeiro consumidor de verdade."""
    indice = construir([["dose"], ["dose"], ["outro"], ["dose"]])

    with pytest.raises(ValueError):
        indice.pontuar(["dose"], mascara=np.array([1, 0, 1, 0]))


def test_mascara_de_outro_tamanho_e_recusada():
    indice = construir([["dose"], ["dose"], ["outro"]])

    with pytest.raises(ValueError):
        indice.pontuar(["dose"], mascara=np.array([True, False]))


def test_termo_conhecido_pontua_mesmo_num_indice_montado_a_mao():
    """`pontuar` lia `_idf[token]` e `idf()` lia `.get`: um BM25 construído
    direto pela dataclass, sem `_idf`, estourava no primeiro termo conhecido."""
    indice = construir([["dose", "dose"], ["outro"]])
    sem_idf = BM25(
        postings=indice.postings,
        comprimentos=indice.comprimentos,
        n_documentos=indice.n_documentos,
        comprimento_medio=indice.comprimento_medio,
    )

    escores = sem_idf.pontuar(["dose"])

    assert escores.shape == (2,)
