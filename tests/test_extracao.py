"""A chamada de referência do PDF desce do sobrescrito e cola na palavra.

No papel é "…quando possível^27,39"; extraído vira "possível27,39". Na maior
parte das vezes o modelo ignora, mas quando a chamada cola num NÚMERO de dose
("4 U^109,111,112" vira "4 U109,111,112") ele pode ler 109 unidades de insulina.
"""

import pytest

from geracao_ancorada.fatiamento.extracao import _limpar, remover_chamadas


def test_chamada_multipla_colada_na_palavra_e_removida():
    assert remover_chamadas(
        "intensidade vigorosa, quando possível27,39. Os exercícios"
    ) == "intensidade vigorosa, quando possível. Os exercícios"


def test_chamada_colada_na_dose_e_removida():
    # O caso perigoso: "4 U" é a dose, 109/111/112 são referências.
    assert remover_chamadas("ajustar 4 U109,111,112. Quando a glicemia") == (
        "ajustar 4 U. Quando a glicemia"
    )
    assert remover_chamadas("acima de 130 mg/dL106,108,109.") == "acima de 130 mg/dL."


def test_chamada_unica_depois_de_palavra_longa_e_removida():
    assert remover_chamadas("lesões na retina73.") == "lesões na retina."
    assert remover_chamadas("insulina glargina115.") == "insulina glargina."
    assert remover_chamadas("em idosos116; a meta") == "em idosos; a meta"


def test_faixa_com_travessao_e_removida():
    assert remover_chamadas("inferiores64,142–144.") == "inferiores."


def test_sigla_clinica_com_numero_e_preservada():
    intacto = "DM2, DM1, iSGLT2, HbA1c e (DPP4) com IMC em kg/m2."
    assert remover_chamadas(intacto) == intacto


def test_url_e_nome_de_arquivo_nao_sao_mutilados():
    intacto = "https://doi.org/10.2337/dc23-S009 e cab36.pdf e s41574-023-00833-4"
    assert remover_chamadas(intacto) == intacto


def test_numero_separado_por_espaco_nao_e_tocado():
    # Em português a vírgula é decimal: "IMC 27,39" é um valor, não citação.
    intacto = "IMC de 27,39 kg/m2 e glicemia de 5,7 mmol/L"
    assert remover_chamadas(intacto) == intacto


def test_a_extracao_limpa_cada_linha_que_le():
    # A regra só serve se rodar na leitura do PDF, não como utilitário solto.
    assert _limpar("  ajustar 4 U109,111,112.  ") == "ajustar 4 U."


def test_endereco_com_numero_no_meio_nao_e_mutilado():
    # A URL é o que permite conferir a fonte; mexer nela é pior que a sujeira.
    for intacto in (
        "https://portalarquivos2.saude.gov.br/images/pdf/manual.pdf",
        "https://intergrowth21.tghn.org/standards/",
        "https://bvsms.saude.gov.br/bvs/saudelegis/2018/Reso588.pdf",
        "http://bit.ly/3dEHeh6",
        "www.cdc.gov/pertussis/surv-manual/chpt10-pertussis.html",
        "https://www.who.int/iris/december2013.pdf?ua=1",
    ):
        assert remover_chamadas(intacto) == intacto


def test_intervalo_de_paginas_nao_e_chamada():
    # "Diabetes Care. 2023;46(Supplement_1):S73–85." — S73–85 é a paginação.
    intacto = "Diabetes Care. 2023;46(Supplement_1):S73–85."
    assert remover_chamadas(intacto) == intacto
    intacto2 = "v. 12, p. 1122.e1–1122.e10, 2020."
    assert remover_chamadas(intacto2) == intacto2


def test_termo_clinico_curto_com_numero_e_preservado():
    # "beta2" e "alfa2" são nomes de receptor, não chamada de referência. Aqui a
    # regra desiste: o custo de mutilar um termo é maior que o de deixar uma
    # chamada passar, e não há sinal no texto que separe "beta2" de "retina73"
    # além do tamanho da palavra.
    for intacto in (
        "agonista beta2 de ação prolongada",
        "long-acting beta2-agonist",
        "bloqueador alfa2 central",
    ):
        assert remover_chamadas(intacto) == intacto


def test_parte_inteira_de_decimal_colado_nao_e_removida():
    # "variação entre0.23 menor" — tirar o 0 deixaria "entre.23". O decimal com
    # VÍRGULA não precisa desta proteção: nos 11 documentos ele sempre aparece
    # com espaço antes ("para 76,0 anos"), e é a cola sem espaço que a regra usa
    # como sinal de chamada.
    assert remover_chamadas("variação entre0.23 menor") == "variação entre0.23 menor"
    assert remover_chamadas("expectativa para 76,0 anos") == "expectativa para 76,0 anos"


@pytest.mark.corpus
def test_texto_corrido_e_lido_fora_da_area_da_tabela():
    """A afirmação central do fatiador, presa numa página conhecida.

    Na página 11 do PCDT de diabete de 2026 mora o Quadro 5, com a linha de
    anamnese "Sexo, Idade, Tabagismo". A página extraída de uma vez embaralha
    a tabela com o parágrafo ao lado; lida fora das áreas de tabela, a linha
    fica só no bloco de tabela.
    """
    from pathlib import Path

    from geracao_ancorada.fatiamento.extracao import extrair

    caminho = Path(__file__).resolve().parents[1] / "fontes" / "cache" / "pcdt-dm2-2026.pdf"
    if not caminho.exists():
        pytest.skip("corpus não baixado")

    pagina = [b for b in extrair(caminho) if b.pagina == 11]
    tabelas = [b for b in pagina if b.tipo == "tabela"]
    textos = [b for b in pagina if b.tipo == "texto"]

    assert any("Tabagismo" in b.texto for b in tabelas)
    assert not any("Tabagismo" in b.texto for b in textos)
