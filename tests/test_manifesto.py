from pathlib import Path

import pytest

from geracao_ancorada.fontes.manifesto import ManifestoInvalido, carregar_manifesto

PCDT_DM2 = """
fontes:
  - id: pcdt-dm2-2024
    titulo: PCDT Diabete Melito Tipo 2
    orgao: MS/CONITEC
    ano: 2024
    url: https://www.gov.br/conitec/pt-br/assuntos/protocolos/PCDTDM2.pdf/@@display-file/file
    base_legal: anexo da Portaria SECTICS/MS no 7, de 28/02/2024
    redistribuivel: true
    areas: [clinica-medica]
"""


def test_carrega_fonte_com_a_procedencia_declarada(tmp_path: Path) -> None:
    caminho = tmp_path / "manifesto.yaml"
    caminho.write_text(PCDT_DM2, encoding="utf-8")

    (fonte,) = carregar_manifesto(caminho)

    assert fonte.id == "pcdt-dm2-2024"
    assert fonte.orgao == "MS/CONITEC"
    assert fonte.ano == 2024
    assert fonte.url.endswith("/@@display-file/file")
    assert fonte.base_legal.startswith("anexo da Portaria")
    assert fonte.redistribuivel is True
    assert fonte.areas == ["clinica-medica"]


SEM_BASE_LEGAL = """
fontes:
  - id: diretriz-sbd-2025
    titulo: Diretriz da Sociedade Brasileira de Diabetes
    orgao: SBD
    ano: 2025
    url: https://diretriz.diabetes.org.br/
    redistribuivel: false
    areas: [clinica-medica]
"""


def test_rejeita_fonte_sem_base_legal_nomeando_a_fonte(tmp_path: Path) -> None:
    caminho = tmp_path / "manifesto.yaml"
    caminho.write_text(SEM_BASE_LEGAL, encoding="utf-8")

    with pytest.raises(ManifestoInvalido) as erro:
        carregar_manifesto(caminho)

    assert "diretriz-sbd-2025" in str(erro.value)
    assert "base_legal" in str(erro.value)


IDS_REPETIDOS = """
fontes:
  - id: cab-32
    titulo: Caderno de Atenção Básica 32 - pré-natal
    orgao: MS
    ano: 2012
    url: https://bvsms.saude.gov.br/cab32.pdf
    base_legal: publicação oficial, reprodução com citação
    redistribuivel: true
    areas: [ginecologia-obstetricia]
  - id: cab-32
    titulo: Caderno de Atenção Básica 32 - outra edição
    orgao: MS
    ano: 2013
    url: https://bvsms.saude.gov.br/cab32-2013.pdf
    base_legal: publicação oficial, reprodução com citação
    redistribuivel: true
    areas: [ginecologia-obstetricia]
"""


def test_rejeita_ids_repetidos_porque_o_id_e_a_chave_de_procedencia(tmp_path: Path) -> None:
    caminho = tmp_path / "manifesto.yaml"
    caminho.write_text(IDS_REPETIDOS, encoding="utf-8")

    with pytest.raises(ManifestoInvalido) as erro:
        carregar_manifesto(caminho)

    assert "cab-32" in str(erro.value)


SUBSTITUICAO = """
fontes:
  - id: pcdt-dm2-2026
    titulo: PCDT Diabete Melito Tipo 2
    orgao: MS/CONITEC
    ano: 2026
    url: https://www.gov.br/conitec/x/@@display-file/file
    base_legal: anexo da Portaria SCTIE/MS no 13, de 21/02/2026
    redistribuivel: true
    areas: [clinica-medica]
    substitui: pcdt-dm2-2024
  - id: pcdt-dm2-2024
    titulo: PCDT Diabete Melito Tipo 2
    orgao: MS/CONITEC
    ano: 2024
    url: https://www.gov.br/conitec/y/@@display-file/file
    base_legal: anexo da Portaria SECTICS/MS no 7, de 28/02/2024
    redistribuivel: true
    areas: [clinica-medica]
"""


def test_registra_qual_documento_a_fonte_substitui(tmp_path: Path) -> None:
    caminho = tmp_path / "manifesto.yaml"
    caminho.write_text(SUBSTITUICAO, encoding="utf-8")

    vigente, anterior = carregar_manifesto(caminho)

    assert vigente.substitui == "pcdt-dm2-2024"
    assert anterior.substitui is None


def test_rejeita_substituicao_de_fonte_que_nao_esta_no_manifesto(tmp_path: Path) -> None:
    """Ponteiro solto quebra a cadeia de vigência que decide o desempate na fusão."""
    caminho = tmp_path / "manifesto.yaml"
    caminho.write_text(SUBSTITUICAO.replace("substitui: pcdt-dm2-2024",
                                            "substitui: pcdt-dm2-2019"), encoding="utf-8")

    with pytest.raises(ManifestoInvalido) as erro:
        carregar_manifesto(caminho)

    assert "pcdt-dm2-2019" in str(erro.value)
