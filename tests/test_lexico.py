"""O tokenizador do braço léxico, testado contra a grafia real do corpus.

Cada teste aqui nasce de uma string que existe nos PDF do manifesto. A
tokenização é onde o domínio quebra: as regras genéricas de recuperação de
texto apagam justamente o que a prova de residência pergunta, que é o número
da lei, a unidade da dose e a sigla de duas letras.
"""

import pytest

from geracao_ancorada.estante.lexico import dobrar, tokenizar, tokenizar_consulta


def test_lei_8080_nao_vira_8_e_080():
    """O ponto de milhar não quebra o token, e a forma sem ponto sai junto.

    O corpus escreve "Lei nº 8.080/1990" e quem responde a prova escreve
    "Lei 8080". Sem a emissão extra a pergunta não encontra a lei.
    """
    tokens = tokenizar("Lei nº 8.080, de 19 de setembro de 1990")

    assert "8.080" in tokens
    assert "8080" in tokens
    assert "8" not in tokens
    assert "080" not in tokens


def test_lei_com_barra_casa_pelo_numero_e_pelo_ano():
    """"Lei nº 8.080/1990" tem que casar com quem escreve só "8080" ou "1990"."""
    tokens = tokenizar("Lei nº 8.080/1990")

    assert "8.080/1990" in tokens
    assert "8.080" in tokens
    assert "8080" in tokens
    assert "1990" in tokens


def test_unidade_com_barra_sobrevive_e_se_abre():
    """mg/dL fica inteiro e também se abre, porque a prova escreve os dois."""
    tokens = tokenizar("HDL-colesterol < 35 mg/dL")

    assert "mg/dl" in tokens
    assert "mg" in tokens
    assert "dl" in tokens
    assert "hdl-colesterol" in tokens
    assert "hdl" in tokens
    assert "colesterol" in tokens


def test_dT_adulto_nao_vira_DT_infantil():
    """A única regra com consequência clínica.

    dT é a dupla adulto e DT é a dupla infantil, vacinas diferentes para
    populações diferentes. Dobrar a caixa sem mais nada funde as duas.
    """
    adulto = tokenizar("reforço com dT a cada dez anos")
    infantil = tokenizar("esquema com DT na infância")

    assert "dT|C" in adulto
    assert "DT|C" in infantil
    assert "dT|C" not in infantil
    assert "DT|C" not in adulto
    # A forma dobrada continua saindo nas duas, porque quem escreve "dt" na
    # pergunta ainda tem que alcançar as duas passagens.
    assert "dt" in adulto and "dt" in infantil


def test_token_de_duas_letras_sobrevive():
    """Sem piso de comprimento. O `if len(t) < 3` de praxe apaga dT inteiro."""
    tokens = tokenizar("dT e BK no escarro")

    assert "dt" in tokens
    assert "bk" in tokens


def test_pipe_de_tabela_nao_conta_como_termo():
    """A tabela chega do fatiador com as células separadas por barra vertical."""
    tokens = tokenizar(" | Fatores de risco | Pontuação | ")

    assert "|" not in tokens
    assert tokens == ["fatores", "de", "risco", "pontuacao"]


def test_acento_nao_separa():
    """O acento dobra, e não quebra a palavra em duas."""
    assert dobrar("Vigilância") == "vigilancia"
    assert tokenizar("notificação compulsória") == ["notificacao", "compulsoria"]


def test_ordinal_dobra_para_a_letra():
    """NFD não decompõe ª nem º, e o corpus está cheio dos dois."""
    assert dobrar("1ª dose") == "1a dose"
    assert dobrar("Art. 1º") == "art. 1o"
    assert "1a" in tokenizar("1ª dose aos dois meses")


def test_sigla_com_hifen_alcanca_a_forma_colada_do_corpus():
    """Conferido nos PDF: o corpus escreve DPP4 colado e nunca com hífen.

    Quem escreve a pergunta escreve "iDPP-4", que é a forma da prova. Sem a
    emissão da forma sem separador as duas grafias nunca se encontram.
    """
    do_corpus = tokenizar("dipeptidil peptidase 4 (DPP4), agonistas do GLP-1")
    da_pergunta = tokenizar("inibidores da DPP-4")

    assert "dpp4" in do_corpus
    assert "dpp4" in da_pergunta
    assert "dpp-4" in da_pergunta
    assert "dpp" in da_pergunta
    assert "glp-1" in do_corpus and "glp" in do_corpus


def test_sem_stemmer_e_de_proposito():
    """Morfologia é serviço do braço denso, e é para isso que o híbrido existe.

    Um stemmer aqui reduziria "vacinas" e "vacinação" ao mesmo radical e o
    braço léxico deixaria de ser o que acha o termo exato.
    """
    assert tokenizar("vacinas") == ["vacinas"]
    assert tokenizar("vacinação") == ["vacinacao"]
    assert tokenizar("vacinas") != tokenizar("vacinação")


def test_indice_nao_remove_palavra_funcional():
    """Sem stoplist no índice. A IDF do Lucene já cuida do termo frequente."""
    assert tokenizar("de acordo com a dose") == ["de", "acordo", "com", "a", "dose"]


def test_consulta_remove_palavra_funcional():
    """Na consulta a stoplist entra, porque o enunciado é longo e cheio dela."""
    tokens = tokenizar_consulta("qual é a dose de acordo com o protocolo")

    assert "dose" in tokens
    assert "protocolo" in tokens
    assert "de" not in tokens
    assert "com" not in tokens
    assert "a" not in tokens


def test_consulta_preserva_o_termo_raro():
    """A stoplist da consulta não pode encostar na sigla que sustenta a busca."""
    tokens = tokenizar_consulta("quando se indica a IGHVZ e a dT")

    assert "ighvz" in tokens
    assert "dt" in tokens
    assert "dT|C" in tokens


@pytest.mark.parametrize(
    "texto,esperado",
    [
        # As três primeiras são grafias conferidas nos PDF do manifesto.
        ("insuficiência renal (TFG menor que 30 mL/min/1,73 m2)", "1,73"),
        ("insuficiência renal (TFG menor que 30 mL/min/1,73 m2)", "ml/min/1,73"),
        ("a taxa de filtração glomerular pela equação CKD-EPI", "ckd-epi"),
        # A quarta é forma, e não citação: é o formato de número de portaria.
        ("Portaria 2.336/2023", "2.336/2023"),
    ],
)
def test_pontuacao_interna_nunca_quebra_o_token(texto, esperado):
    assert esperado in tokenizar(texto)
