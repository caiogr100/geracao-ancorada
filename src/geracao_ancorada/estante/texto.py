"""O texto que cada braço indexa, que não é o mesmo nos dois.

O cabeçalho faz efeitos OPOSTOS, e o desenho antigo tratava isso como decisão
única. No denso ele paga: é o que sustenta o pedaço órfão e a tabela cujo texto
sozinho é " | Anamnese | - Sexo - Idade". No léxico ele estraga de três jeitos:
"Protocolo", "Diabete" e "MS/CONITEC" passam a aparecer em todos os pedaços
daquela fonte, com IDF perto de zero, consumindo frequência sem discriminar;
entram no comprimento do documento, e com b=0,75 a normalização pune o pedaço
curto, que no corpus é a tabela, que é onde mora a dose; e o cabeçalho carrega o
número de seção, e a tokenização preserva "8.3.4.2" como termo, plantando
numeral de seção no mesmo espaço onde se procura "8.080".

A página sai dos dois braços. Ela não diz nada sobre o conteúdo e vira o token
"25", que colide com dose e com idade em milhares de pedaços. Por isso a moldura
é montada aqui e não sai de `Pedaco.cabecalho`, que existe para a CITAÇÃO e
imprime a página de propósito.
"""

from geracao_ancorada.fatiamento.pedacos import Pedaco


def _moldura(p: Pedaco) -> str:
    """A procedência do pedaço em uma linha, sem a página.

    Montada por junção e não por f-string, porque 9% dos pedaços do corpus não
    têm rótulo de seção e o Guia de Vigilância não numera as suas: a f-string
    cega sairia como "Guia (2024) —  — . " e o traço órfão viraria token.
    """
    secao = ". ".join(parte for parte in (p.secao, p.titulo_secao) if parte)
    campos = [c for c in (p.parte, secao) if c]
    return " — ".join([f"{p.titulo_fonte} ({p.ano})", *campos])


def texto_indexado_denso(p: Pedaco, legenda: str = "") -> str:
    """Moldura, legenda e texto — o que vai para o vetor."""
    return "\n".join(filter(None, [_moldura(p), legenda, p.texto.strip()]))


def texto_indexado_lexico(p: Pedaco, legenda: str = "") -> str:
    """Só a legenda e o texto — o que vai para o BM25."""
    return f"{legenda}\n{p.texto}".strip()
