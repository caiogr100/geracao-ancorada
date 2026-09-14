"""O que vai para cada braço, que não é a mesma coisa nos dois.

O cabeçalho faz efeitos opostos. No denso ele paga: é o que sustenta a tabela
cujo texto sozinho é " | Anamnese | - Sexo - Idade". No léxico ele estraga:
"Protocolo" e "MS/CONITEC" passariam a aparecer em todos os pedaços daquela
fonte, com IDF perto de zero, consumindo frequência sem discriminar, e entram no
comprimento do documento, onde a normalização do BM25 pune justamente o pedaço
curto — que no corpus é a tabela, que é onde mora a dose.

A página sai dos dois. Ela não diz nada sobre o conteúdo e viraria o token "25",
que colide com dose e com idade em milhares de pedaços.
"""

import pytest

from geracao_ancorada.estante.texto import texto_indexado_denso, texto_indexado_lexico
from geracao_ancorada.fatiamento.pedacos import Pedaco

LEGENDA_5 = "Quadro 5. Avaliação clínica e laboratorial para a estimativa de risco cardiovascular."


def pedaco(**troca):
    base = dict(
        id="pcdt-dm2-2026#anexo/5.5.4",
        fonte_id="pcdt-dm2-2026",
        titulo_fonte="Protocolo Clínico e Diretrizes Terapêuticas do Diabete Melito tipo 2",
        orgao="MS/CONITEC",
        ano=2026,
        parte="ANEXO",
        secao="5.5",
        titulo_secao="Estratificação do risco cardiovascular",
        pagina_inicial=11,
        pagina_final=11,
        tipo="tabela",
        texto=" | Anamnese | \n- Sexo - Idade - Tabagismo",
    )
    base.update(troca)
    return Pedaco(**base)


def test_pagina_nao_entra_no_texto_indexado():
    """Nem no denso nem no léxico, e a página 11 deste pedaço é o caso."""
    p = pedaco()

    assert "11" not in texto_indexado_denso(p).split("\n")[0]
    assert "p. 11" not in texto_indexado_denso(p)
    assert "p. 11" not in texto_indexado_lexico(p)


def test_moldura_entra_no_denso_e_nao_entra_no_lexico():
    p = pedaco()

    denso = texto_indexado_denso(p)
    lexico = texto_indexado_lexico(p)

    assert "MS/CONITEC" not in lexico and "Protocolo Clínico" not in lexico
    assert "Estratificação do risco cardiovascular" in denso
    assert lexico == p.texto.strip()


def test_legenda_entra_nos_dois_bracos():
    """É para isso que o módulo da legenda existe: a tabela ganha o nome."""
    p = pedaco()

    assert LEGENDA_5 in texto_indexado_denso(p, LEGENDA_5)
    assert texto_indexado_lexico(p, LEGENDA_5).startswith("Quadro 5.")


def test_o_texto_do_pedaco_continua_inteiro():
    p = pedaco()

    assert p.texto.strip() in texto_indexado_denso(p)


def test_pedaco_sem_secao_nao_gera_pontuacao_solta():
    """São 431 pedaços do corpus sem rótulo de seção, 9% do total.

    A moldura montada com f-string cega sairia como "Guia (2024) —  — . ",
    e esse traço órfão vira token e entra no vetor.
    """
    p = pedaco(parte="", secao="", titulo_secao="", ano=2024,
               titulo_fonte="Guia de Vigilância")

    cabecalho = texto_indexado_denso(p).split("\n")[0]

    assert cabecalho == "Guia de Vigilância (2024)"


def test_secao_sem_numero_sai_so_com_o_titulo():
    """O Guia não numera seção: o rótulo dele é "} PERÍODO DE INCUBAÇÃO"."""
    p = pedaco(
        titulo_fonte="Guia de Vigilância em Saúde, volume 1",
        ano=2024,
        parte="DOENÇA MENINGOCÓCICA",
        secao="",
        titulo_secao="PERÍODO DE INCUBAÇÃO",
    )

    cabecalho = texto_indexado_denso(p).split("\n")[0]

    assert cabecalho == (
        "Guia de Vigilância em Saúde, volume 1 (2024) — DOENÇA MENINGOCÓCICA "
        "— PERÍODO DE INCUBAÇÃO"
    )


@pytest.mark.corpus
def test_nenhum_texto_indexado_do_corpus_tem_moldura_quebrada(pedacos_do_corpus):
    """A varredura que pega o separador órfão nos 4.499 pedaços.

    Ela NÃO exige que o cabeçalho deixe de terminar em ponto: no termo de
    esclarecimento do PCDT de hipertensão o título da seção é uma frase
    inteira, "10. Informar ao usuário o valor de PA obtido na medição.", e o
    ponto é dele. A primeira versão deste teste reprovava esse caso, ou seja,
    media coisa diferente do que o nome dela diz.
    """
    for p in pedacos_do_corpus:
        cabecalho = texto_indexado_denso(p).split("\n")[0]
        assert not cabecalho.rstrip().endswith("—"), (p.id, cabecalho)
        assert "—  —" not in cabecalho, (p.id, cabecalho)
        assert " — ." not in cabecalho, (p.id, cabecalho)
        assert not cabecalho.endswith(" ."), (p.id, cabecalho)
