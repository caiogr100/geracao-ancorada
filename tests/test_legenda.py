"""A legenda que o fatiador deixou no pedaço de texto vizinho.

O PCDT escreve a legenda numa linha e a tabela logo adiante, e o fatiador
separa as duas. A tabela do risco cardiovascular do PCDT de diabete entra no
índice como " | Anamnese | \n- Sexo - Idade - Tabagismo", sem conter "Quadro 5"
nem uma palavra do que ela é. Este módulo devolve a legenda para ela.

É dívida declarada, e o número está medido: das 1.167 tabelas indexáveis do
corpus, só **163 (14%)** têm alguma legenda na própria seção. O lugar certo do
conserto é o fatiador. O que este módulo não faz, ele não esconde.
"""

import pytest

from geracao_ancorada.estante.legenda import legenda
from geracao_ancorada.fatiamento.pedacos import Pedaco

# Copiados letra por letra de pcdt-dm2-2026#anexo/5.5.*, que é o caso do desenho.
LEGENDA_5 = "Quadro 5. Avaliação clínica e laboratorial para a estimativa de risco cardiovascular."
CHAMADA_EM_PROSA = "O Quadro 5 apresenta os dados relevantes da anamnese, do exame físico"
TABELA_5 = " | Anamnese | \n- Sexo - Idade - Tabagismo - História prévia de doença"


def pedaco(sufixo, tipo="texto", texto="", secao="5.5", parte="ANEXO"):
    return Pedaco(
        id=f"pcdt-dm2-2026#{parte.lower()}/{sufixo}",
        fonte_id="pcdt-dm2-2026",
        titulo_fonte="PCDT Diabete Melito tipo 2",
        orgao="MS/CONITEC",
        ano=2026,
        parte=parte,
        secao=secao,
        titulo_secao="Estratificação do risco cardiovascular",
        tipo=tipo,
        texto=texto,
    )


def test_tabela_carrega_o_quadro_que_a_nomeia():
    texto = pedaco("5.5.1", texto=f"{CHAMADA_EM_PROSA}\n{LEGENDA_5}\n")
    tabela = pedaco("5.5.4", tipo="tabela", texto=TABELA_5)

    assert legenda(tabela, [texto, tabela]) == LEGENDA_5


def test_chamada_em_prosa_nao_e_legenda():
    """"O Quadro 5 apresenta..." é referência no meio da frase, e está no corpus.

    O que separa uma da outra é a pontuação logo depois do número, e é por
    isso que ela é exigida.
    """
    texto = pedaco("5.5.1", texto=CHAMADA_EM_PROSA)
    tabela = pedaco("5.5.4", tipo="tabela", texto=TABELA_5)

    assert legenda(tabela, [texto, tabela]) == ""


def test_legenda_depois_da_tabela_nao_vale():
    """A legenda nomeia o que vem DEPOIS dela; a de baixo é de outra tabela."""
    tabela = pedaco("5.5.2", tipo="tabela", texto=TABELA_5)
    seguinte = pedaco("5.5.3", texto="Quadro 6. Metas no tratamento do diabete melito tipo 2.")

    assert legenda(tabela, [tabela, seguinte]) == ""


def test_a_legenda_mais_proxima_antes_e_a_que_vale():
    """Uma seção com duas tabelas tem duas legendas, e a de cima é da de cima."""
    primeira = pedaco("8.5.1", texto="Quadro 11. Estratégia de tratamento da hiperglicemia.")
    tabela_um = pedaco("8.5.2", tipo="tabela", texto=" | esquema | ")
    segunda = pedaco("8.5.3", texto="Quadro 12. Classificação das neuropatias diabéticas.")
    tabela_dois = pedaco("8.5.4", tipo="tabela", texto=" | grau | ")
    vizinhos = [primeira, tabela_um, segunda, tabela_dois]

    assert legenda(tabela_um, vizinhos).startswith("Quadro 11.")
    assert legenda(tabela_dois, vizinhos).startswith("Quadro 12.")


def test_chamada_de_referencia_grudada_no_numero_nao_vira_legenda():
    """"Quadro 634." é o "Quadro 6" com a chamada "34" colada pela extração.

    Existe uma no corpus, e sem o limite de dois dígitos ela venceria a
    legenda verdadeira do Quadro 6, que vem logo depois.
    """
    texto = pedaco("8.1.1", texto="Quadro 634. Valores de HbA1c < 7,0% apresentaram maior risco")
    tabela = pedaco("8.1.2", tipo="tabela", texto=" | meta | ")

    assert legenda(tabela, [texto, tabela]) == ""


def test_legenda_de_figura_nao_nomeia_tabela():
    """Medido no índice: 55 das 115 legendas atribuídas começavam com "Figura",
    e a amostra é toda de tabela de verdade com nome de fluxograma. A tabela de
    critérios diagnósticos do PCDT de diabete saía como "Figura 1. Fluxograma
    de rastreamento", porque a legenda da figura vem antes dela na seção."""
    figura = pedaco("5.3.1", secao="5.3", texto="Figura 1. Fluxograma de rastreamento e diagnóstico do diabete melito tipo 2.")
    tabela = pedaco("5.3.5", secao="5.3", tipo="tabela", texto=" | Critérios | Normal | Pré-diabetes | DM2")

    assert legenda(tabela, [figura, tabela]) == ""


def test_legenda_de_outra_secao_nao_atravessa():
    de_fora = pedaco("5.4.1", secao="5.4", texto="Quadro 4. Critérios diagnósticos.")
    tabela = pedaco("5.5.4", tipo="tabela", texto=TABELA_5)

    assert legenda(tabela, [de_fora, tabela]) == ""


def test_tabela_sem_legenda_nenhuma_devolve_vazio():
    """São 1.004 das 1.167 tabelas do corpus, e o módulo não inventa nada."""
    tabela = pedaco("5.5.4", tipo="tabela", texto=TABELA_5)

    assert legenda(tabela, [tabela]) == ""


@pytest.mark.corpus
def test_a_tabela_do_risco_cardiovascular_recebe_o_quadro_5(pedacos_do_corpus):
    """O caso do desenho, no corpus de verdade e não na fixtura."""
    alvo = next(
        p for p in pedacos_do_corpus if p.id == "pcdt-dm2-2026#anexo/5.5.4"
    )

    achada = legenda(alvo, pedacos_do_corpus)

    assert achada.startswith("Quadro 5.")
    assert "risco cardiovascular" in achada
