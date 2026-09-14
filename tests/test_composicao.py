"""O que entra no índice principal e o que fica de fora.

Dois cortes, e os dois existem porque o erro que eles evitam sai impresso na
citação. O primeiro é a fonte superada: o PCDT de diabete de 2024 e o de 2026
dizem coisas diferentes sobre a mesma insulina, e com os dois no mesmo índice o
gerador cita norma revogada com página. O segundo é o apêndice de metodologia,
cuja tabela de conflito de interesse do PCDT de 2026 traz "Recebeu honorários
para apresentação de aulas" — texto que casa com consulta sobre tratamento e
não responde nada de clínica.

As fixturas são rótulos LITERAIS, copiados dos onze PDF depois de medidos, e o
que elas provam está no teste do corpus no fim do arquivo.
"""

from pathlib import Path

import pytest

from geracao_ancorada.estante.composicao import indexaveis, superadas
from geracao_ancorada.fatiamento.pedacos import Pedaco
from geracao_ancorada.fontes.manifesto import carregar_manifesto

MANIFESTO = Path(__file__).resolve().parents[1] / "fontes" / "manifesto.yaml"


def pedaco(fonte_id, parte="", titulo_secao="", texto="texto qualquer", secao=""):
    return Pedaco(
        id=f"{fonte_id}#{parte}/{secao or '1.1'}",
        fonte_id=fonte_id,
        titulo_fonte="",
        orgao="MS",
        ano=2026,
        parte=parte,
        secao=secao,
        titulo_secao=titulo_secao,
        texto=texto,
    )


def test_superadas_sai_do_manifesto_de_verdade():
    """O campo `substitui` já existe e ninguém o lia."""
    fontes = carregar_manifesto(MANIFESTO)

    assert superadas(fontes) == frozenset({"pcdt-dm2-2024"})


def test_pcdt_superado_nao_entra_no_indice_principal():
    pedacos = [
        pedaco("pcdt-dm2-2024", texto="insulina NPH em duas aplicações"),
        pedaco("pcdt-dm2-2026", texto="insulina NPH em duas aplicações"),
    ]

    sobrevivem = indexaveis(pedacos, superadas=frozenset({"pcdt-dm2-2024"}))

    assert [p.fonte_id for p in sobrevivem] == ["pcdt-dm2-2026"]


def test_superado_volta_quando_a_sonda_contrafactual_pede():
    """A colisão 2024/2026 é o estrato que a régua precisa medir de propósito."""
    pedacos = [pedaco("pcdt-dm2-2024"), pedaco("pcdt-dm2-2026")]

    sobrevivem = indexaveis(
        pedacos, superadas=frozenset({"pcdt-dm2-2024"}), incluir_superadas=True
    )

    assert len(sobrevivem) == 2


def test_apendice_de_metodologia_fica_de_fora():
    """O rótulo do PCDT de 2026, copiado letra por letra do PDF."""
    metodologia = "APÊNDICE 1 – METODOLOGIA DE BUSCA E AVALIAÇÃO DA LITERATURA"
    pedacos = [
        pedaco("pcdt-dm2-2026", parte="", texto="alvo de hemoglobina glicada"),
        pedaco(
            "pcdt-dm2-2026",
            parte=metodologia,
            titulo_secao="Equipe de elaboração e partes interessadas",
            texto="Participante | Conflitos de interesses declarados |",
        ),
    ]

    sobrevivem = indexaveis(pedacos)

    assert [p.parte for p in sobrevivem] == [""]


def test_apendice_de_metodologia_sem_o_titulo_no_rotulo_tambem_fica_de_fora():
    """Metade dos PCDT rotula o apêndice só com o número.

    No de asma, de dislipidemia e de DPOC o rótulo é "APÊNDICE 1" pelado, e o
    que diz que aquilo é metodologia está no título da seção ou na primeira
    linha do texto. São 57 pedaços, e a regra que olha só o rótulo os deixa
    todos entrar.
    """
    pedacos = [
        pedaco(
            "pcdt-asma-2021",
            parte="APÊNDICE 1",
            titulo_secao="Equipe de elaboração e partes interessadas",
        ),
        pedaco(
            "pcdt-dpoc-2025",
            parte="APÊNDICE 1",
            texto="METODOLOGIA DE BUSCA E AVALIAÇÃO DA LITERATURA PARA ATUALIZAÇÃO",
        ),
    ]

    assert indexaveis(pedacos) == []


def test_o_apendice_inteiro_sai_junto_com_a_parte_que_o_denuncia():
    """A marca de metodologia está em UMA seção e o apêndice tem cinco.

    No PCDT de DPOC só a seção 5 se chama "Equipe de elaboração"; as outras
    quatro são "Escopo e finalidade", "Consulta pública" e afins, que sozinhas
    não denunciam nada. Decidir pedaço a pedaço deixaria quatro quintos do
    apêndice dentro do índice.
    """
    apendice = [
        pedaco("pcdt-dpoc-2025", parte="APÊNDICE 1", titulo_secao="Consulta pública"),
        pedaco(
            "pcdt-dpoc-2025",
            parte="APÊNDICE 1",
            titulo_secao="Equipe de elaboração e partes interessadas",
        ),
    ]

    assert indexaveis(apendice) == []


def test_apendice_clinico_continua_no_indice():
    """O corte é o da metodologia, e não o do apêndice.

    O escore de Framingham do PCDT de dislipidemia e as quatro páginas de
    aplicação de insulina do de diabete são conteúdo de prova, e moram em
    apêndice.
    """
    pedacos = [
        pedaco(
            "pcdt-dislipidemia-2019",
            parte="APÊNDICE 2 – Escore de risco de Framingham",
            texto="Idade (em anos) Pontos 20-34 -9",
        ),
        pedaco(
            "pcdt-dm2-2024",
            parte="APÊNDICE A – ORIENTAÇÃO EM RELAÇÃO À APLICAÇÃO DE INSULINA",
        ),
    ]

    assert len(indexaveis(pedacos)) == 2


def test_equipe_multiprofissional_no_corpo_nao_derruba_a_secao():
    """"Equipe" sozinha é palavra de clínica, e o PCDT a usa o tempo todo."""
    pedacos = [
        pedaco(
            "pcdt-has-2025",
            parte="",
            titulo_secao="Tratamento não farmacológico",
            texto="acompanhamento pela equipe multiprofissional da atenção básica",
        )
    ]

    assert len(indexaveis(pedacos)) == 1


@pytest.mark.corpus
def test_nenhuma_declaracao_de_conflito_de_interesse_sobrevive(pedacos_do_corpus):
    """A varredura sobre os onze PDF, que é onde o corte é medido de verdade."""
    fontes = carregar_manifesto(MANIFESTO)

    sobrevivem = indexaveis(pedacos_do_corpus, superadas=superadas(fontes))

    ids = {p.id for p in sobrevivem}
    assert "pcdt-dm2-2026#ap-ndice-1-metodologia-d/5.5" not in ids
    assert not [p for p in sobrevivem if p.fonte_id == "pcdt-dm2-2024"]
    assert not [
        p for p in sobrevivem if "Conflitos de interesses declarados" in p.texto
    ]
