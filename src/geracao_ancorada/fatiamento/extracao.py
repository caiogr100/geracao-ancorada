"""Extração do PDF preservando a tabela como texto e a página de origem.

A tabela é o caso difícil e é onde mora metade da informação de um PCDT: dose,
faixa etária, critério. Extrair a página inteira como texto corrido embaralha a
tabela com o parágrafo ao lado. Aqui as duas coisas saem separadas, e o texto
corrido é lido *fora* das áreas de tabela.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

# Marcadores de lista em fonte Symbol vêm como área de uso privado do Unicode.
_GLIFOS_PRIVADOS = re.compile(r"[-]")


@dataclass(frozen=True)
class Bloco:
    """Um pedaço bruto de página, antes de virar unidade de recuperação."""

    pagina: int
    tipo: str  # "texto" ou "tabela"
    texto: str


# A chamada de referência é sobrescrita no PDF e desce colada na palavra:
# "…quando possível^27,39" sai como "possível27,39". Quase sempre o modelo
# ignora, mas quando ela cola num número de dose ("4 U^109,111,112" vira "4
# U109,111,112") ele pode ler 109 unidades de insulina. Duas regras, as duas
# exigindo que o número esteja COLADO — com espaço antes seria valor de verdade,
# porque em português a vírgula é decimal ("IMC 27,39").
#
# A: duas ou mais referências, e a vírgula entre elas é obrigatória — o
#    travessão só entra dentro de uma faixa ("64,142–144"). Sem a vírgula o que
#    se pega é a paginação de uma referência ("...(Supplement_1):S73–85").
#    Nenhuma sigla clínica tem essa forma, então basta vir depois de letra.
_CHAMADA_MULTIPLA = re.compile(r"(?<=[A-Za-zÀ-ÿ)])\d+(?:–\d+)?(?:,\d+(?:–\d+)?)+(?!\w)")

# B: uma referência só, e aí é preciso distinguir de sigla ("DM2", "iSGLT2",
#    "kg/m2", "HbA1c") e de URL ("dc23-S009", "cab36.pdf"). O que separa: a
#    chamada vem depois de palavra corrida — CINCO letras ou mais, a última
#    minúscula — e encerra o token sem abrir um decimal.
#
#    Cinco e não quatro por causa de "beta2" e "alfa2", que são nome de receptor
#    e não citação. Nada no texto separa "beta2" de "retina73" além do tamanho
#    da palavra, então a regra desiste do que tem quatro letras: deixar passar
#    uma chamada custa menos que mutilar um termo clínico. O preço são casos
#    como "fumo75" e "Sede1", que ficam sujos.
_CHAMADA_UNICA = re.compile(r"(?<=[A-Za-zÀ-ÿ]{4}[a-zà-ÿ])\d+(?![\w.,]\d)(?!\w)")


# Endereço e nome de arquivo têm número no meio por natureza
# ("portalarquivos2.saude.gov.br", "Reso588.pdf", "chpt10-pertussis.html") e são
# justamente o que permite conferir a fonte. Ali não se mexe.
_ENDERECO = re.compile(
    r"https?://|www\.|\.[A-Za-z]{2,5}(?:/|\?|$)|\.(?:com|org|net|gov|edu|int|br|pdf|html?)\b",
    re.IGNORECASE,
)


def remover_chamadas(texto: str) -> str:
    """Tira a chamada de referência que a extração colou no fim da palavra."""
    return " ".join(
        token if _ENDERECO.search(token)
        else _CHAMADA_UNICA.sub("", _CHAMADA_MULTIPLA.sub("", token))
        for token in texto.split(" ")
    )


def _limpar(texto: str) -> str:
    texto = _GLIFOS_PRIVADOS.sub("-", texto or "")
    texto = texto.replace("\xa0", " ")
    texto = remover_chamadas(texto)
    return re.sub(r"[ \t]+", " ", texto).strip()


def _tabela_para_texto(tabela: list[list[str | None]]) -> str:
    linhas = []
    for linha in tabela:
        celulas = [_limpar(c or "").replace("\n", " ") for c in linha]
        if any(celulas):
            linhas.append(" | ".join(celulas))
    return "\n".join(linhas)


def _fora_das_tabelas(pagina, caixas):
    """Devolve a página sem os objetos que caem dentro de uma tabela."""
    if not caixas:
        return pagina

    def manter(obj):
        x = (obj["x0"] + obj["x1"]) / 2
        y = (obj["top"] + obj["bottom"]) / 2
        return not any(x0 <= x <= x1 and topo <= y <= base for x0, topo, x1, base in caixas)

    return pagina.filter(manter)


def extrair(caminho: Path) -> list[Bloco]:
    """Lê o PDF e devolve os blocos de cada página, texto e tabela separados."""
    blocos: list[Bloco] = []
    with pdfplumber.open(caminho) as pdf:
        for numero, pagina in enumerate(pdf.pages, start=1):
            tabelas = pagina.find_tables()
            caixas = [t.bbox for t in tabelas]

            corrido = _fora_das_tabelas(pagina, caixas).extract_text() or ""
            for linha in corrido.split("\n"):
                limpa = _limpar(linha)
                if limpa:
                    blocos.append(Bloco(pagina=numero, tipo="texto", texto=limpa))

            for tabela in tabelas:
                texto = _tabela_para_texto(tabela.extract())
                if texto:
                    blocos.append(Bloco(pagina=numero, tipo="tabela", texto=texto))
    return blocos
