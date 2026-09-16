"""A busca híbrida sobre uma estante pequena montada à mão.

Seis pedaços com vetores de três dimensões bastam para exercitar o que a
busca precisa acertar: o filtro é máscara e não muda o escore de quem
sobrevive, a lista léxica para no escore zero, os escores crus de cada braço
viajam no achado, e o registro grava todo candidato com os dois escores.
"""

import json

import numpy as np
import pytest

from geracao_ancorada.estante import bm25, lexico
from geracao_ancorada.estante.busca import Achado, Filtro, buscar, registrar
from geracao_ancorada.estante.indice import Estante
from geracao_ancorada.estante.texto import texto_indexado_lexico
from geracao_ancorada.fatiamento.pedacos import Pedaco


def _pedaco(id_, texto, *, areas, ano=2024, fonte="f1", tipo="texto", descartavel=False):
    return Pedaco(
        id=id_, fonte_id=fonte, titulo_fonte="Fonte", orgao="MS", ano=ano,
        areas=areas, texto=texto, tipo=tipo, descartavel=descartavel,
    )


def _unitario(*coordenadas):
    v = np.asarray(coordenadas, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def estante():
    pedacos = [
        _pedaco("calendario#1", "vacina varicela IGHVZ gestante", areas=["pediatria", "preventiva"]),
        _pedaco("asma#1", "corticoide inalatório dose", areas=["clinica-medica", "pediatria"], fonte="f2"),
        _pedaco("dm2#1", "metformina dose máxima", areas=["clinica-medica"], fonte="f3", ano=2026),
        _pedaco("dm2#2", "insulina esquema basal", areas=["clinica-medica"], fonte="f3", ano=2026),
        _pedaco("tb#1", "tuberculose esquema básico", areas=["clinica-medica", "preventiva"], fonte="f4"),
        _pedaco("tb#ref", "referências bibliográficas varicela", areas=["clinica-medica", "preventiva"],
                fonte="f4", descartavel=True),
    ]
    matriz = np.stack([
        _unitario(1, 0, 0),
        _unitario(0, 1, 0),
        _unitario(0, 0, 1),
        _unitario(0, 1, 1),
        _unitario(1, 1, 0),
        _unitario(1, 0, 1),
    ])
    indice_lexico = bm25.construir([lexico.tokenizar(texto_indexado_lexico(p)) for p in pedacos])
    return Estante(pedacos=pedacos, legendas=[""] * 6, matriz=matriz, bm25=indice_lexico, assinatura=None)


def _vetor_fixo(vetor):
    def vetorizar(texto):
        return vetor
    return vetorizar


def test_filtro_por_area_e_intersecao_e_nao_igualdade(estante):
    """Pediatria alcança o calendário (pediatria+preventiva) e a asma (clínica+pediatria)."""
    mascara = Filtro(areas=("pediatria",)).mascara(estante.pedacos)

    assert mascara.dtype == np.bool_
    assert [p.id for p, m in zip(estante.pedacos, mascara) if m] == ["calendario#1", "asma#1"]


def test_descartavel_nao_aparece_por_padrao(estante):
    achados = buscar(estante, "varicela", vetor=_unitario(1, 0, 1))

    assert "tb#ref" not in [a.id for a in achados]
    assert "calendario#1" in [a.id for a in achados]


def test_descartavel_entra_quando_pedido(estante):
    achados = buscar(estante, "varicela", vetor=_unitario(1, 0, 1),
                     filtro=Filtro(incluir_descartaveis=True))

    assert "tb#ref" in [a.id for a in achados]


def test_lexico_para_no_escore_zero(estante):
    """Quem não casa token nenhum não ganha posição léxica, nem ponto na fusão."""
    achados = buscar(estante, "IGHVZ", vetor=_unitario(0, 0, 1))

    por_id = {a.id: a for a in achados}
    assert por_id["calendario#1"].posicao_lexica == 1
    assert por_id["calendario#1"].bm25 > 0
    assert por_id["dm2#1"].posicao_lexica is None
    assert por_id["dm2#1"].bm25 == 0
    assert por_id["dm2#1"].rrf == pytest.approx(1 / 61)


def test_termo_raro_entra_pelo_braco_lexico_com_garantia(estante):
    """A sigla é primeira no léxico e última no denso; pela fusão ela fica de
    fora dos dois entregues, e entra mesmo assim, no fim, com a marca dizendo por quê."""
    achados = buscar(estante, "IGHVZ dose esquema", vetor=_unitario(0, 1, 1), n=2)

    por_fusao = sorted(achados.candidatos, key=lambda a: -a.rrf)
    assert "calendario#1" not in [a.id for a in por_fusao[:2]]
    assert [a.id for a in achados] == ["dm2#2", "calendario#1"]
    assert [a.garantido for a in achados] == [False, True]
    assert achados[1].posicao_lexica == 1
    assert achados[1].posicao_densa == 5


def test_achado_carrega_os_escores_crus_de_cada_braco(estante):
    vetor = _unitario(0, 1, 1)
    achados = buscar(estante, "insulina esquema", vetor=vetor)

    por_id = {a.id: a for a in achados}
    tokens = lexico.tokenizar_consulta("insulina esquema")
    assert por_id["dm2#2"].cosseno == pytest.approx(float(estante.matriz[3] @ vetor))
    assert por_id["dm2#2"].bm25 == pytest.approx(float(estante.bm25.pontuar(tokens)[3]))
    assert por_id["dm2#2"].posicao_densa == 1


def test_filtro_nao_muda_o_escore_de_quem_sobrevive(estante):
    vetor = _unitario(0, 1, 1)
    sem = {a.id: a for a in buscar(estante, "esquema", vetor=vetor)}
    com = {a.id: a for a in buscar(estante, "esquema", vetor=vetor, filtro=Filtro(fontes=("f3",)))}

    assert set(com) == {"dm2#1", "dm2#2"}
    for id_, achado in com.items():
        assert achado.cosseno == pytest.approx(sem[id_].cosseno)
        assert achado.bm25 == pytest.approx(sem[id_].bm25)


def test_vetor_pronto_nao_chama_o_vetorizador(estante):
    def explode(texto):
        raise AssertionError("não devia vetorizar")

    achados = buscar(estante, "esquema", vetor=_unitario(0, 1, 1), vetorizar=explode)

    assert achados


def test_sem_vetor_o_vetorizador_e_chamado_com_o_texto(estante):
    chamado = []

    def vetorizar(texto):
        chamado.append(texto)
        return _unitario(0, 1, 1)

    buscar(estante, "esquema", vetorizar=vetorizar)

    assert chamado == ["esquema"]


def test_empate_dentro_do_braco_e_resolvido_por_id(estante):
    """Dois pedaços com o mesmo cosseno saem sempre na mesma ordem."""
    achados = buscar(estante, "nada", vetor=_unitario(0, 1, 1))

    # dm2#2 tem cosseno 1; asma#1, dm2#1, tb#1 e calendario#1 vêm depois; tb#1 e asma#1
    # empatam entre si (ambos 1/sqrt(2)) e a ordem é a alfabética.
    posicoes = {a.id: a.posicao_densa for a in achados}
    assert posicoes["asma#1"] < posicoes["tb#1"]


def test_registrar_grava_todos_os_candidatos_com_escore_de_cada_braco(estante, tmp_path):
    caminho = tmp_path / "corrida.jsonl"
    achados = buscar(estante, "esquema", vetor=_unitario(0, 1, 1), n=2)

    registrar("esquema", achados, caminho)
    registrar("esquema", achados, caminho)

    linhas = [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines()]
    assert len(linhas) == 2
    corrida = linhas[0]
    assert corrida["consulta"] == "esquema"
    assert corrida["n"] == 2
    assert corrida["constante_rrf"] == 60
    assert len(corrida["candidatos"]) == len(achados.candidatos)
    primeiro = corrida["candidatos"][0]
    assert {"id", "sha256_texto", "posicao_densa", "posicao_lexica", "cosseno", "bm25", "rrf", "garantido"} <= set(primeiro)
    assert "texto" not in primeiro


def test_entrega_sao_os_n_primeiros_e_candidatos_sao_todos(estante):
    achados = buscar(estante, "esquema", vetor=_unitario(0, 1, 1), n=2)

    assert len(achados) == 2
    assert len(achados.candidatos) == 5
    assert list(achados) == achados.candidatos[:2]


def test_reordenador_de_brinquedo_inverte_a_lista(estante):
    def inverter(consulta, candidatos):
        return list(reversed(candidatos))

    normal = buscar(estante, "esquema", vetor=_unitario(0, 1, 1), n=2)
    invertida = buscar(estante, "esquema", vetor=_unitario(0, 1, 1), n=2, reordenador=inverter)

    assert [a.id for a in invertida] == [a.id for a in reversed(normal.candidatos)][:2]


# --- o dado real --------------------------------------------------------------


@pytest.mark.corpus
@pytest.mark.gpu
def test_a_sigla_rara_entra_pelo_braco_lexico_no_indice_real():
    """"IGHVZ" é o caso que motiva a garantia: o tokenizador do modelo denso
    estilhaça a sigla, e só o braço léxico a encontra em primeiro."""
    from geracao_ancorada.estante import indice
    from geracao_ancorada.estante.indice import carregar

    if not (indice.REGISTRO / "assinatura.json").exists():
        pytest.skip("índice não construído")

    achados = buscar(carregar(), "IGHVZ")

    primeiro_lexico = next(a for a in achados.candidatos if a.posicao_lexica == 1)
    assert primeiro_lexico.id.startswith("calendario-vacinacao-2026#17.")
    assert primeiro_lexico.posicao_densa is None or primeiro_lexico.posicao_densa > 1
    assert primeiro_lexico.id in [a.id for a in achados]
    assert all(a.bm25 > 0 for a in achados.candidatos if a.posicao_lexica is not None)
    assert all(a.posicao_lexica is None for a in achados.candidatos if a.bm25 == 0)


@pytest.mark.corpus
@pytest.mark.gpu
def test_filtro_por_pediatria_alcanca_o_calendario_e_o_pcdt_de_asma_no_indice_real():
    from geracao_ancorada.estante import indice
    from geracao_ancorada.estante.indice import carregar

    if not (indice.REGISTRO / "assinatura.json").exists():
        pytest.skip("índice não construído")

    estante = carregar()
    fontes = {p.fonte_id for p, m in zip(estante.pedacos, Filtro(areas=("pediatria",)).mascara(estante.pedacos)) if m}

    assert {"calendario-vacinacao-2026", "pcdt-asma-2021"} <= fontes
    assert "pcdt-dislipidemia-2019" not in fontes
