"""O braço denso, e o contrato com o Ollama que decide se ele mente.

O risco deste módulo não é errar a conta, é acertar a conta sobre um texto que
chegou pela metade. Com `truncate: true` o Ollama corta o que passa do teto e
devolve 200 com um vetor de aparência normal, e uma tabela cortada no meio vira
um pedaço que existe no índice, pontua, e não contém a linha da dose. Por isso
o pedido sai com `truncate: false` e o 400 vira erro nomeando o pedaço.
"""

import numpy as np
import pytest

from geracao_ancorada.estante.vetores import (
    DIMENSAO,
    TextoLongoDemais,
    vetorizar_consulta,
    vetorizar_corpus,
    vetorizar_documentos,
)


def ollama_falso(dimensao=DIMENSAO, valor=3.0):
    """Devolve vetores NÃO normalizados, para que a normalização seja testada."""
    chamadas = []

    def chamar(corpo):
        chamadas.append(corpo)
        n = len(corpo["input"])
        return 200, {"embeddings": [[valor] * dimensao for _ in range(n)]}

    chamar.chamadas = chamadas
    return chamar


def test_vetor_sai_normalizado():
    """O índice compara por produto interno, e isso só é cosseno se a norma é 1."""
    chamar = ollama_falso()

    matriz = vetorizar_documentos(["dose de reforço"], chamar=chamar)

    assert np.allclose(np.linalg.norm(matriz, axis=1), 1.0, atol=1e-3)


def test_matriz_sai_float32_com_as_colunas_do_modelo():
    chamar = ollama_falso()

    matriz = vetorizar_documentos(["um", "dois", "três"], chamar=chamar)

    assert matriz.dtype == np.float32
    assert matriz.shape == (3, DIMENSAO)


def test_ordem_de_entrada_e_preservada_entre_lotes():
    """A posição na matriz é o identificador do pedaço em todo o caminho."""
    def chamar(corpo):
        # A marca é a DIREÇÃO do vetor, e não o tamanho dele: a normalização
        # apaga o tamanho, e uma primeira versão deste teste marcava o tamanho
        # e não enxergava troca de ordem nenhuma.
        vetores = []
        for t in corpo["input"]:
            v = [0.0] * DIMENSAO
            v[len(t)] = 1.0
            vetores.append(v)
        return 200, {"embeddings": vetores}

    matriz = vetorizar_documentos(["a", "bb", "ccc", "dddd", "eeeee"], lote=2, chamar=chamar)

    assert [int(np.argmax(linha)) for linha in matriz] == [1, 2, 3, 4, 5]


def test_pedido_sai_sempre_com_truncate_falso():
    """A guarda inteira depende deste campo, e ele não é opcional."""
    chamar = ollama_falso()

    vetorizar_documentos(["um", "dois"], lote=1, chamar=chamar)

    assert all(c["truncate"] is False for c in chamar.chamadas)


def test_vetorizar_pede_descarregar_no_ultimo_lote():
    """Sem isso o bge-m3 fica residente e o gerador entra em cima dele."""
    chamar = ollama_falso()

    vetorizar_documentos(["um", "dois", "três"], lote=1, chamar=chamar)

    assert [c["keep_alive"] for c in chamar.chamadas][-1] == "0"
    assert all(c["keep_alive"] != "0" for c in chamar.chamadas[:-1])


def test_consulta_nao_descarrega_o_modelo():
    """Descarregar a cada pergunta faria o modelo recarregar a cada pergunta."""
    chamar = ollama_falso()

    vetorizar_consulta("qual a dose de reforço", chamar=chamar)

    assert chamar.chamadas[0]["keep_alive"] != "0"


def test_consulta_sai_com_um_vetor_normalizado_e_nao_com_uma_matriz():
    chamar = ollama_falso()

    v = vetorizar_consulta("qual a dose de reforço", chamar=chamar)

    assert v.shape == (DIMENSAO,)
    assert np.allclose(np.linalg.norm(v), 1.0, atol=1e-3)


def test_recusa_do_ollama_vira_erro_nomeando_o_texto():
    """O 400 do `truncate: false` precisa dizer QUAL pedaço estourou."""
    def chamar(corpo):
        return 400, {"error": "the input length exceeds the context length"}

    with pytest.raises(TextoLongoDemais) as erro:
        vetorizar_documentos(["tabela enorme do termo de esclarecimento"], chamar=chamar)

    assert "tabela enorme" in str(erro.value)


def test_vetor_nulo_e_recusado_em_vez_de_virar_nan():
    """Normalizar zero dá divisão por zero, e o NaN contaminaria a busca inteira."""
    chamar = ollama_falso(valor=0.0)

    with pytest.raises(ValueError):
        vetorizar_documentos(["texto"], chamar=chamar)


def test_dimensao_inesperada_e_recusada():
    """O multilingual-e5-large também tem 1024, então conferir isto não basta,
    mas um modelo de 384 passaria calado e destruiria o índice."""
    chamar = ollama_falso(dimensao=384)

    with pytest.raises(ValueError):
        vetorizar_documentos(["texto"], chamar=chamar)


@pytest.mark.gpu
def test_texto_acima_do_teto_e_recusado_e_nao_cortado():
    """O único teste contra o servidor de verdade, e ele só separa "funcionou"
    de "cortou" se o texto passa do teto.

    A primeira versão trocava a última frase de um texto de 1.400 caracteres e
    conferia que o vetor mudava. O teto medido é de cerca de 8.900 em prosa,
    então nada era cortado de nenhum jeito, e o teste ficava verde com
    `truncate: true`. Com `truncate: false`, o que se observa acima do teto é
    a recusa, e não um vetor diferente.
    """
    prosa = "Metas terapêuticas de hemoglobina glicada no diabete melito tipo 2. " * 200
    assert len(prosa) > 12_000

    with pytest.raises(TextoLongoDemais):
        vetorizar_consulta(prosa)


def test_erro_nomeia_o_texto_culpado_e_nao_o_primeiro_do_lote():
    """O pedido vai em lote de 16, e o 400 vem do lote inteiro.

    Sem isolar, a mensagem acusaria o primeiro texto do lote, que na prática é
    um pedaço inocente, e a caça ao culpado seria manual.
    """
    culpado = "tabela do termo de esclarecimento com seis mil caracteres"

    def chamar(corpo):
        if any(t == culpado for t in corpo["input"]):
            return 400, {"error": "the input length exceeds the context length"}
        return 200, {"embeddings": [[1.0] * DIMENSAO for _ in corpo["input"]]}

    with pytest.raises(TextoLongoDemais) as erro:
        vetorizar_documentos(["curto", culpado, "outro curto"], lote=8, chamar=chamar)

    assert "tabela do termo" in str(erro.value)
    assert "curto" not in str(erro.value)


def test_vetorizar_corpus_separa_os_recusados_em_vez_de_abortar():
    """A indexação inteira não pode morrer por causa de um pedaço grande.

    Quem chama `vetorizar_documentos` quer o erro alto, porque uma consulta
    recusada é um defeito. A indexação quer a lista: ela anota quem ficou de
    fora, segue com o resto, e o motivo entra na assinatura do índice.
    """
    grande = "tabela do termo de esclarecimento"

    def chamar(corpo):
        if any(t == grande for t in corpo["input"]):
            return 400, {"error": "the input length exceeds the context length"}
        return 200, {"embeddings": [[1.0] * DIMENSAO for _ in corpo["input"]]}

    saida = vetorizar_corpus(["um", grande, "dois", "três"], lote=2, chamar=chamar)

    assert saida.recusados == [1]
    assert saida.aceitos == [0, 2, 3]
    assert saida.matriz.shape == (3, DIMENSAO)


def test_vetorizar_corpus_sem_recusa_nenhuma_devolve_tudo_na_ordem():
    chamar = ollama_falso()

    saida = vetorizar_corpus(["um", "dois", "três"], lote=2, chamar=chamar)

    assert saida.recusados == []
    assert saida.aceitos == [0, 1, 2]
    assert saida.matriz.shape == (3, DIMENSAO)


def test_vetorizar_corpus_descarrega_o_modelo_no_fim():
    chamar = ollama_falso()

    vetorizar_corpus(["um", "dois", "três"], lote=1, chamar=chamar)

    assert chamar.chamadas[-1]["keep_alive"] == "0"


def _servidor_que_cai_no_segundo_lote():
    """Aceita o primeiro lote, devolve 500 no segundo, aceita o resto."""
    chamadas = []

    def chamar(corpo):
        chamadas.append(corpo)
        if len(chamadas) == 2:
            return 500, {"error": "o servidor caiu"}
        return 200, {"embeddings": [[1.0] * DIMENSAO for _ in corpo["input"]]}

    chamar.chamadas = chamadas
    return chamar


def test_vetorizar_corpus_descarrega_o_modelo_mesmo_quando_o_servidor_falha():
    """O descarregamento existe para o gerador não entrar em cima de um modelo
    residente. Um erro que não seja o 400 saía da função com o modelo na placa,
    que é justamente o caminho em que ninguém está olhando."""
    chamar = _servidor_que_cai_no_segundo_lote()

    with pytest.raises(RuntimeError):
        vetorizar_corpus(["um", "dois", "três", "quatro", "cinco", "seis"], lote=2, chamar=chamar)

    assert chamar.chamadas[-1]["keep_alive"] == "0"


def test_vetorizar_documentos_descarrega_o_modelo_mesmo_quando_o_servidor_falha():
    """A mesma garantia na porta da consulta em lote, que amarrava o
    descarregamento ao último lote e não chegava nele quando um anterior caía."""
    chamar = _servidor_que_cai_no_segundo_lote()

    with pytest.raises(RuntimeError):
        vetorizar_documentos(["um", "dois", "três", "quatro", "cinco", "seis"], lote=2, chamar=chamar)

    assert chamar.chamadas[-1]["keep_alive"] == "0"


def test_lote_com_um_recusado_custa_um_pedido_por_texto_e_nao_mais():
    """O 400 vem do lote inteiro, e o isolamento do culpado acontecia duas
    vezes: uma para montar a mensagem de erro, que a indexação descarta, e
    outra na própria indexação. Um lote de 16 com um recusado custava 34
    pedidos, e cada texto inocente era vetorizado três vezes."""
    grande = "tabela do termo de esclarecimento"
    chamadas = []

    def chamar(corpo):
        chamadas.append(corpo)
        if any(t == grande for t in corpo["input"]):
            return 400, {"error": "the input length exceeds the context length"}
        return 200, {"embeddings": [[1.0] * DIMENSAO for _ in corpo["input"]]}

    vetorizar_corpus(["um", grande, "dois", "três"], lote=4, chamar=chamar)

    # O lote, um pedido por texto do lote, e o descarregamento.
    assert len(chamadas) == 1 + 4 + 1
