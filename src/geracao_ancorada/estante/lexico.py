"""Tokenização do português normativo, para o braço léxico da busca.

O braço léxico existe para achar o termo exato que o braço denso erra: o
número da lei, a sigla da vacina, a unidade da dose. As regras genéricas de
recuperação de texto trabalham contra isso. O piso de três caracteres apaga
"dT"; o stemmer funde "vacinas" com "vacinação"; e quebrar o token na
pontuação transforma a Lei 8.080 em "8" e "080", que casam com qualquer
coisa.

Nenhuma das regras abaixo foi escolhida por gosto. Cada uma responde a uma
grafia conferida nos PDF do manifesto, e o teste correspondente carrega a
string de onde ela saiu.
"""

import re
import unicodedata

VERSAO = "1"
"""Versão do tokenizador. Entra na assinatura do índice.

Mudar qualquer regra daqui muda o vocabulário e invalida a estatística do
BM25 gravada. A assinatura é o que impede carregar um índice construído com
outra versão e medir a diferença errada.
"""

# O NFD decompõe o acento, e não o indicador ordinal: "1ª" continua "1ª"
# depois da normalização. O corpus está cheio dos dois ("1ª dose" no
# Calendário, "Art. 1º" na portaria que abre cada PCDT), então o mapa é
# explícito.
_ORDINAIS = str.maketrans({"ª": "a", "º": "o"})

# Ponto, vírgula, barra e hífen não quebram o token. São eles que seguram
# "8.080", "1,73", "mg/dL" e "CKD-EPI", que é exatamente o que a prova
# pergunta. Fora do token, qualquer outro caractere separa, inclusive a barra
# vertical com que o fatiador entrega as células da tabela.
_TOKEN = re.compile(r"[0-9a-z]+(?:[./,\-][0-9a-z]+)*")

_SEPARADOR_INTERNO = re.compile(r"[-/]")

# O ponto de milhar, e não o ponto decimal. Em português o decimal é a
# vírgula, então o ponto entre dígitos com exatamente três à direita é
# separador de milhar: "8.080" abre para "8080", e "1,73" fica intacto.
_MILHAR = re.compile(r"(?<=\d)\.(?=\d{3}(?:\D|$))")

_CAIXA_ATE = 5
"""Comprimento até o qual o token com maiúscula emite a forma com caixa.

É a regra com consequência clínica: dT é a dupla adulto e DT é a dupla
infantil, vacinas diferentes para populações diferentes. Acima de cinco
caracteres a distinção deixa de compensar o posting a mais.
"""

_MARCA_DE_CAIXA = "|C"

# Stoplist da CONSULTA, e só dela. A consulta é o enunciado da questão com o
# comando e as alternativas, que é longo e quase todo funcional. No índice não
# entra stoplist nenhuma: a IDF do Lucene já resolve o termo frequente, e uma
# lista fixa apagaria o "a" de "vitamina A".
_FUNCIONAIS = frozenset(
    """a o as os um uma uns umas de do da dos das e ou em no na nos nas por
    para com sem sob sobre ao aos que qual quais quando onde como se ser sao
    foi era seu sua seus suas este esta esse essa isso aquilo entre ate apos
    antes mais menos muito pouco tambem ja nao sim""".split()
)


def dobrar(texto: str) -> str:
    """Tira o acento, resolve o ordinal e baixa a caixa."""
    decomposto = unicodedata.normalize("NFD", texto.translate(_ORDINAIS))
    sem_acento = "".join(c for c in decomposto if not unicodedata.combining(c))
    return sem_acento.casefold()


def _dobrar_com_origem(texto: str) -> tuple[str, list[int]]:
    """Dobra o texto guardando, para cada caractere da saída, o índice de onde
    ele veio.

    Serve a uma coisa só: recuperar o token com a caixa ORIGINAL depois de
    casar o padrão sobre o texto dobrado. Fatiar o texto original pelos
    índices do dobrado só funciona enquanto o comprimento não muda, e ele
    muda (o casefold de "ß" são dois caracteres). Aqui não depende disso.
    """
    dobrado: list[str] = []
    origem: list[int] = []
    for posicao, caractere in enumerate(texto):
        convertido = dobrar(caractere)
        dobrado.append(convertido)
        origem.extend([posicao] * len(convertido))
    return "".join(dobrado), origem


def _abrir(token: str) -> list[str]:
    """As formas extras de um token, sempre com o inteiro à frente.

    Três aberturas, e cada uma tem uma grafia do corpus por trás: a forma sem
    o ponto de milhar, porque o corpus escreve "Lei nº 8.080" e a pergunta
    escreve "Lei 8080"; os segmentos de hífen e de barra, porque o corpus
    escreve "mg/dL" e a pergunta escreve "mg"; e a forma sem separador
    nenhum, porque o corpus escreve "DPP4" e a prova escreve "iDPP-4".

    O inteiro nunca sai da lista. Abrir sem manter o inteiro trocaria uma
    recuperação exata por uma aproximada, que é o oposto do que este braço faz.
    """
    formas = [token]

    def acrescentar(forma: str) -> None:
        if forma and forma not in formas:
            formas.append(forma)

    acrescentar(_MILHAR.sub("", token))

    segmentos = [s for s in _SEPARADOR_INTERNO.split(token) if s]
    if len(segmentos) > 1:
        for segmento in segmentos:
            acrescentar(segmento)
            acrescentar(_MILHAR.sub("", segmento))
        acrescentar(_SEPARADOR_INTERNO.sub("", token))

    return formas


def tokenizar(texto: str) -> list[str]:
    """Os tokens de um pedaço, como entram no índice.

    Sem piso de comprimento, sem stemmer e sem stoplist, os três declarados e
    testados. A ordem da saída acompanha a do texto, e a primeira forma de
    cada token é sempre a inteira.
    """
    dobrado, origem = _dobrar_com_origem(texto)
    saida: list[str] = []

    for achado in _TOKEN.finditer(dobrado):
        saida.extend(_abrir(achado.group()))
        bruto = texto[origem[achado.start()]: origem[achado.end() - 1] + 1]
        if len(bruto) <= _CAIXA_ATE and any(c.isupper() for c in bruto):
            saida.append(f"{bruto}{_MARCA_DE_CAIXA}")

    return saida


def tokenizar_consulta(texto: str) -> list[str]:
    """Os tokens de uma consulta: os mesmos, menos as palavras funcionais."""
    return [t for t in tokenizar(texto) if t not in _FUNCIONAIS]
