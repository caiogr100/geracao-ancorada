import hashlib
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from geracao_ancorada.fontes.aquisicao import (
    ConteudoInesperado,
    FonteDivergente,
    adquirir,
)
from geracao_ancorada.fontes.manifesto import Fonte

PDF = b"%PDF-1.7\nglicemia plasmatica aleatoria >= 200 mg/dL\n"

PCDT = Fonte(
    id="pcdt-dm2-2024",
    titulo="PCDT Diabete Melito Tipo 2",
    orgao="MS/CONITEC",
    ano=2024,
    url="https://www.gov.br/conitec/PCDTDM2.pdf/@@display-file/file",
    base_legal="anexo da Portaria SECTICS/MS no 7, de 28/02/2024",
    redistribuivel=True,
    areas=["clinica-medica"],
)


def baixador_fixo(conteudo: bytes):
    def baixar(url: str) -> bytes:
        return conteudo

    return baixar


def test_registra_hash_tamanho_e_data_do_que_foi_baixado(tmp_path: Path) -> None:
    registro = adquirir(PCDT, destino=tmp_path, baixar=baixador_fixo(PDF))

    assert registro.sha256 == hashlib.sha256(PDF).hexdigest()
    assert registro.bytes == len(PDF)
    assert registro.baixado_em == date.today().isoformat()
    assert registro.url == PCDT.url
    assert (tmp_path / "pcdt-dm2-2024.pdf").read_bytes() == PDF


def test_acusa_divergencia_quando_a_fonte_muda_sob_o_snapshot(tmp_path: Path) -> None:
    congelado = replace(PCDT, sha256=hashlib.sha256(PDF).hexdigest())

    with pytest.raises(FonteDivergente) as erro:
        adquirir(congelado, destino=tmp_path, baixar=baixador_fixo(b"%PDF-1.7\noutra coisa\n"))

    assert "pcdt-dm2-2024" in str(erro.value)


def test_recusa_html_no_lugar_do_pdf(tmp_path: Path) -> None:
    """A URL divulgada do CONITEC devolve a página, não o arquivo; sem o
    sufixo /@@display-file/file o corpus seria montado sobre HTML."""
    pagina = b"<!DOCTYPE html>\n<html lang=\"pt-br\"><head><title>PCDT</title>"

    with pytest.raises(ConteudoInesperado) as erro:
        adquirir(PCDT, destino=tmp_path, baixar=baixador_fixo(pagina))

    assert "pcdt-dm2-2024" in str(erro.value)
    assert not (tmp_path / "pcdt-dm2-2024.pdf").exists()
