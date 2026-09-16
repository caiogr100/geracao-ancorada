"""A fusão das duas listas por posição, com o primeiro de cada braço garantido.

Os testes aqui são a aritmética da decisão de 15/09 virada em asserção: com a
constante 60, o primeiro colocado de um braço só vale 1/61 e perde para o
consenso mediano, e a garantia existe para que ele entre mesmo assim.
"""

import pytest

from geracao_ancorada.estante.fusao import (
    CONSTANTE_RRF,
    GARANTIDOS_POR_BRACO,
    aplicar_garantia,
    rrf,
)


def test_constantes_declaradas():
    assert CONSTANTE_RRF == 60
    assert GARANTIDOS_POR_BRACO == 1


def test_pedaco_nas_duas_listas_soma_as_duas_posicoes():
    fundida = rrf({"lexico": ["a", "b"], "denso": ["b", "a"]})

    escores = dict(fundida)
    assert escores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert escores["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_ausente_contribui_zero_e_nao_posicao_imputada():
    """Quem não está numa lista não ganha 1/(k + tamanho + 1) dela."""
    fundida = rrf({"lexico": ["so-lexico"], "denso": ["so-denso", "outro"]})

    escores = dict(fundida)
    assert escores["so-lexico"] == pytest.approx(1 / 61)
    assert escores["outro"] == pytest.approx(1 / 62)


def test_braco_vazio_nao_derruba_a_fusao():
    fundida = rrf({"lexico": [], "denso": ["a", "b"]})

    assert [id_ for id_, _ in fundida] == ["a", "b"]


def test_empate_e_resolvido_por_id():
    """Dois PCDT quase idênticos empatam; a ordem tem de ser a mesma sempre."""
    fundida = rrf({"lexico": ["zeta", "alfa"], "denso": ["alfa", "zeta"]})

    assert [id_ for id_, _ in fundida] == ["alfa", "zeta"]


def test_consenso_mediano_vence_o_primeiro_isolado_sem_garantia():
    """A aritmética que motiva a garantia: 1/61 perde para 1/63 + 1/64."""
    listas = {
        "lexico": ["sigla", "x1", "m1", "m2"],
        "denso": ["d1", "d2", "m1", "m2"],
    }

    fundida = rrf(listas)

    ordem = [id_ for id_, _ in fundida]
    assert ordem.index("m1") < ordem.index("sigla")
    assert ordem.index("m2") < ordem.index("sigla")


def test_garantia_poe_o_primeiro_de_cada_braco_na_entrega():
    """O pedaço que só o léxico acha, em primeiro, entra entre os entregues."""
    lexico = ["sigla"] + [f"l{i}" for i in range(49)]
    denso = [f"d{i}" for i in range(50)]
    consenso = [f"c{i}" for i in range(10)]
    listas = {"lexico": lexico[:1] + consenso + lexico[1:], "denso": denso[:1] + consenso + denso[1:]}
    fundida = rrf(listas)
    assert "sigla" not in [id_ for id_, _ in fundida[:8]]

    entrega = aplicar_garantia(listas, fundida, n=8)

    ids = [id_ for id_, _, _ in entrega]
    assert len(ids) == 8
    assert "sigla" in ids
    assert "d0" in ids
    assert ("sigla", pytest.approx(1 / 61), True) in entrega


def test_a_garantia_decide_quem_entra_e_a_fusao_decide_a_ordem():
    """O garantido não passa na frente de quem a fusão pôs acima dele."""
    lexico = ["sigla"] + [f"l{i}" for i in range(49)]
    denso = [f"d{i}" for i in range(50)]
    consenso = [f"c{i}" for i in range(10)]
    listas = {"lexico": lexico[:1] + consenso + lexico[1:], "denso": denso[:1] + consenso + denso[1:]}
    fundida = rrf(listas)

    entrega = aplicar_garantia(listas, fundida, n=8)

    escores = [escore for _, escore, _ in entrega]
    assert escores == sorted(escores, reverse=True)
    assert [id_ for id_, _, _ in entrega][:6] == consenso[:6]
    assert {id_ for id_, _, _ in entrega[6:]} == {"sigla", "d0"}


def test_garantia_nao_duplica_quem_ja_entrou():
    """O mesmo pedaço em primeiro nos dois braços ocupa uma vaga só."""
    listas = {"lexico": ["a", "b", "c"], "denso": ["a", "c", "b"]}
    fundida = rrf(listas)

    entrega = aplicar_garantia(listas, fundida, n=2)

    assert [id_ for id_, _, _ in entrega] == ["a", "b"]


def test_garantido_que_ja_estava_na_entrega_nao_e_marcado():
    """A marca diz que a garantia mudou a entrega, e não que o pedaço é primeiro."""
    listas = {"lexico": ["a", "b"], "denso": ["a", "b"]}
    fundida = rrf(listas)

    entrega = aplicar_garantia(listas, fundida, n=2)

    assert [g for _, _, g in entrega] == [False, False]


def test_entrega_menor_que_n_quando_ha_menos_candidatos():
    listas = {"lexico": ["a"], "denso": []}
    fundida = rrf(listas)

    entrega = aplicar_garantia(listas, fundida, n=8)

    assert [id_ for id_, _, _ in entrega] == ["a"]
