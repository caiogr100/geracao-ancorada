"""Onde cortar: a junta natural do documento, não um número de caracteres.

Num PCDT a junta é o título numerado. O problema é distinguir o título de
verdade da linha que só começa com número: uma referência bibliográfica
("23. SBD. Diretriz..."), um passo de procedimento ("3. Homogeneizar a
insulina") ou uma quebra de parágrafo infeliz ("4 U109,111,112. Quando a
glicemia...").

A regra que separa os dois não é tamanho nem caixa alta: é que **a numeração de
seção avança em sequência**. Depois de 8.3.4.1 só pode vir 8.3.4.2, 8.3.5, 8.4,
9 ou 8.3.4.1.1. Uma referência numerada quebra a sequência e é descartada.
"""

import re
from dataclasses import dataclass

# A família CORRENTE de PCDT numera com ponto ("8.1. Metas terapêuticas"), e o
# ponto é o que separa o título da nota de rodapé ("1 HC-UFMG, Centro de
# telessaúde..."). Aqui o título pode ser longo: no Calendário de Vacinação ele
# passa de 90 caracteres.
_CANDIDATO = re.compile(r"^(\d+(?:\.\d+)*)\.\s+([A-Za-zÁÂÃÀÉÊÍÓÔÕÚÜÇáâãàéêíóôõúüç].*)$")

# A família ANTIGA de PCDT numera sem ponto ("1 INTRODUÇÃO", "2 CLASSIFICAÇÃO
# ESTATÍSTICA..."). Sem o ponto, o que separa o título da nota de rodapé ("1
# HC-UFMG. Centro de telessaúde") e do parágrafo quebrado é a CAIXA ALTA: o
# título inteiro sem uma minúscula e sem ponto final.
_CANDIDATO_CAIXA_ALTA = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-ZÀ-ÖØ-Þ][^a-zà-ÿ.]{3,88})$")

# O cabeçalho NUMERADO da lista de referências. Precisa ser reconhecido por si,
# e não só quando vira seção aceita: no pcdt-dpoc-2025 um marco falso reinicia a
# numeração logo antes de "11. REFERÊNCIAS", a numerada é então rejeitada, e a
# lista de referências — que começa em "1." — assume a espinha do documento.
#
# O número no cabeçalho é exigido de propósito. Ele é o que indica documento de
# espinha numerada, cuja lista de referências também é numerada e por isso pode
# ser confundida com seção. O "REFERÊNCIAS" pelado do Guia tem regra própria
# (`abre_referencias`, no fim do arquivo), porque lá o que ele suspende é
# outra coisa: a suspensão termina no capítulo seguinte, e não no fim do
# documento. Exige também a linha inteira: "Referências excluídas" de um
# fluxograma não conta.
_INICIO_REFERENCIAS = re.compile(
    r"^\d+(?:\.\d+)*\.?\s+"
    r"(?:REFER[EÊ]NCIAS(?:\s+BIBLIOGR[ÁA]FICAS)?|BIBLIOGRAFIA)\s*:?$",
    re.IGNORECASE,
)


def inicia_referencias(linha: str) -> bool:
    """A linha é o cabeçalho numerado da lista de referências?"""
    return bool(_INICIO_REFERENCIAS.match(linha.strip()))


# Seções cujo conteúdo não é clínico e não deve concorrer na busca.
_DESCARTAVEIS = re.compile(r"^\s*(REFER[EÊ]NCIAS|BIBLIOGRAFIA)\b", re.IGNORECASE)

# Marcos de parte: anexo, apêndice, termo. Reiniciam a numeração de seção.
_MARCO = re.compile(
    r"^\s*(ANEXO|AP[EÊ]NDICE|TERMO DE ESCLARECIMENTO|FICHA FARMACOTERAP[EÊ]UTICA)\b.*$",
    re.IGNORECASE,
)

# O divisor de parte do manual do MS é uma página que traz só "PARTE IV". Tem
# que ser a linha INTEIRA e em caixa alta: no sumário a mesma palavra vem com o
# título e o número da página colados ("PARTE IV • ESTRATÉGIAS PROGRAMÁTICAS"),
# e no texto corrido ela aparece no meio da frase ("Parte II (diagnóstico), em
# que..."). Sem este marco o rótulo do último anexo atravessava a parte inteira.
_DIVISOR_DE_PARTE = re.compile(r"^PARTE\s+[IVXL]+$")


@dataclass(frozen=True)
class Junta:
    """Um título de seção aceito, com a numeração já normalizada."""

    numero: str
    titulo: str
    nivel: tuple[int, ...]

    @property
    def descartavel(self) -> bool:
        return bool(_DESCARTAVEIS.match(self.titulo))


def _nivel(numero: str) -> tuple[int, ...]:
    return tuple(int(parte) for parte in numero.split("."))


def sucede(anterior: tuple[int, ...] | None, candidato: tuple[int, ...]) -> bool:
    """O candidato continua a numeração de seção a partir do anterior?

    Aceita três movimentos, e só eles: descer um nível (5 -> 5.1), avançar no
    mesmo nível (5.1 -> 5.2) e subir para o próximo irmão de qualquer ancestral
    (8.3.4.3 -> 8.4, ou -> 9).
    """
    if anterior is None:
        return candidato == (1,)

    if candidato == anterior + (1,):
        return True

    for corte in range(len(anterior), 0, -1):
        if candidato == anterior[:corte - 1] + (anterior[corte - 1] + 1,):
            return True
    return False


def detectar(linha: str, anterior: tuple[int, ...] | None) -> Junta | None:
    """Devolve a junta se a linha for um título que continua a sequência."""
    limpa = linha.strip()
    achado = _CANDIDATO.match(limpa) or _CANDIDATO_CAIXA_ALTA.match(limpa)
    if achado is None:
        return None

    numero, titulo = achado.group(1), achado.group(2).strip()
    nivel = _nivel(numero)
    if not sucede(anterior, nivel):
        return None
    return Junta(numero=numero, titulo=titulo, nivel=nivel)


# O guia/manual do MS é livro de CAPÍTULOS e não tem espinha numerada: o que abre
# um capítulo é o título em caixa alta seguido da linha "CID-10:". São 64 desses
# nos três volumes do Guia de Vigilância. O par é o que separa o título da
# menção em prosa ("...afecções do Capítulo X / CID-10, exceção para..."), que
# vem sem os dois-pontos e depois de linha com minúscula.
#
# O código não precisa ABRIR a linha. A influenza sazonal abre com "Influenza
# devida a vírus não identificado CID-10: J11", com o código no fim: era o
# único capítulo perdido nos três volumes, e atrás dele o rótulo do anexo
# anterior atravessava 41 páginas. São os dois-pontos que seguram a regra — a
# menção em prosa vem com vírgula ("CID-10, exceção para...").
_CID = re.compile(r"CID[\s-]?10\s*:", re.IGNORECASE)

# O título que não coube numa linha continua na seguinte, e a conjunção é o que
# denuncia a continuação: "TOXOPLASMOSE ADQUIRIDA NA GESTAÇÃO" / "E TOXOPLASMOSE
# CONGÊNITA". Sem ela, duas linhas vizinhas em caixa alta são duas coisas —
# "OUTRAS MENINGITES" agrupa, "MENINGITES BACTERIANAS" é o capítulo.
_CONTINUACAO = re.compile(r"^(?:E|OU)\s+")


def _e_titulo(linha: str) -> bool:
    """Linha curta, em caixa alta e com letra: a forma de um título aqui."""
    return bool(
        linha
        and len(linha) <= 90
        and not re.search(r"[a-zà-ÿ]", linha)
        and re.search(r"[A-ZÀ-ÖØ-Þ]", linha)
    )


def capitulo(linha: str, seguinte: str, anterior: str = "") -> str | None:
    """Título de capítulo de guia, confirmado pela linha de CID que vem depois."""
    limpa = linha.strip()
    if not limpa or len(limpa) > 90 or re.search(r"[a-zà-ÿ]", limpa):
        return None
    if not _CID.search(seguinte.strip()):
        return None

    antes = anterior.strip()
    if _CONTINUACAO.match(limpa) and _e_titulo(antes) and len(antes) + len(limpa) <= 90:
        return f"{antes} {limpa}"
    return limpa


# O Guia marca a subseção com uma chave e caixa alta — "} PERÍODO DE INCUBAÇÃO",
# "} MODO DE TRANSMISSÃO" —, 1.325 delas nos três volumes, e são o eixo em que a
# prova de residência pergunta. A mesma chave marca item de lista comum
# ("} Coletar e registrar os dados de vacinação."): é a caixa alta que separa.
_SUBTITULO = re.compile(r"^\}\s*(.+)$")


def subtitulo(linha: str) -> str | None:
    """Subseção do guia: a linha marcada com a chave, em caixa alta."""
    achado = _SUBTITULO.match(linha.strip())
    if achado is None:
        return None
    titulo = achado.group(1).strip()
    return titulo if _e_titulo(titulo) else None


# O cabeçalho da bibliografia do guia é o "REFERÊNCIAS" PELADO, sem número, 72
# vezes nos três volumes — e por isso escapava da regra numerada acima. O que
# vem depois dele é citação ABNT, e citação não é estrutura: foi assim que a
# linha "Anexo V – Sistema Nacional de Vigilância Epidemiológica (SNVE)", miolo
# de uma referência à Portaria de Consolidação n.º 4, virou rótulo de parte.
# Exige a linha inteira e em caixa alta, senão "Referências excluídas" de um
# fluxograma descartaria o capítulo junto.
_REFERENCIAS_PELADAS = re.compile(
    r"^(?:REFER[EÊ]NCIAS(?:\s+BIBLIOGR[ÁA]FICAS)?|BIBLIOGRAFIA)$"
)


def abre_referencias(linha: str) -> bool:
    """A linha é o cabeçalho sem número da bibliografia?"""
    return bool(_REFERENCIAS_PELADAS.match(linha.strip()))


def titulo_sem_marca(linha: str) -> str | None:
    """Título em caixa alta e sem marcador — o que abre uma seção GERAL.

    As seções gerais do guia (farmacovigilância, saúde do trabalhador, saúde
    ambiental, vigilância do óbito) não usam a chave nem têm CID. Aqui isso
    serve para uma coisa só: FECHAR a bibliografia da seção anterior. Sem
    fechar por ele, a bibliografia atravessava a seção seguinte inteira — 11
    páginas de farmacovigilância saíram como bibliografia.

    O ponto final no fim é o que separa o título da sobra de citação que ficou
    em caixa alta ("SMS, 2007."). Quatro caracteres de piso deixam de fora o
    "SIM"/"NÃO" solto de fluxograma.
    """
    limpa = linha.strip()
    if len(limpa) < 4 or limpa.endswith(".") or not _e_titulo(limpa):
        return None
    return limpa


def marco(linha: str) -> str | None:
    """Título de parte (anexo, apêndice, termo), que reinicia a numeração.

    É o que impede a lista de referências de ser lida como continuação das
    seções: depois de REFERÊNCIAS a detecção fica suspensa até o próximo marco.
    """
    limpa = linha.strip()
    if _DIVISOR_DE_PARTE.match(limpa):
        return limpa
    if len(limpa) > 90 or not _MARCO.match(limpa):
        return None
    return limpa
