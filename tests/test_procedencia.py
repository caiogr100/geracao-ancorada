import json
from pathlib import Path

from geracao_ancorada.fontes.aquisicao import Registro
from geracao_ancorada.fontes.procedencia import escrever_procedencia

REGISTROS = [
    Registro(id="pcdt-dm2-2024", url="https://a/1.pdf", sha256="b" * 64, bytes=1600000,
             baixado_em="2026-09-01"),
    Registro(id="cab-32", url="https://a/2.pdf", sha256="a" * 64, bytes=900000,
             baixado_em="2026-09-01"),
]


def test_grava_procedencia_ordenada_por_id_para_o_diff_ser_legivel(tmp_path: Path) -> None:
    caminho = tmp_path / "procedencia.json"

    escrever_procedencia(REGISTROS, caminho)

    conteudo = json.loads(caminho.read_text(encoding="utf-8"))
    assert [f["id"] for f in conteudo["fontes"]] == ["cab-32", "pcdt-dm2-2024"]
    assert conteudo["fontes"][1]["sha256"] == "b" * 64
    assert caminho.read_text(encoding="utf-8").endswith("\n")
