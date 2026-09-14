"""A legenda que ficou no pedaço de texto vizinho da tabela.

Dívida declarada, e o lugar certo do conserto é o fatiador. O PCDT escreve
"Quadro 5. Avaliação clínica e laboratorial para a estimativa de risco
cardiovascular." numa linha e desenha a tabela adiante; o fatiador separa as
duas, e a tabela entra no índice como " | Anamnese | \n- Sexo - Idade -
Tabagismo", sem conter "Quadro 5" nem uma palavra do que ela é.

O alcance foi medido, e é pequeno: das 1.167 tabelas indexáveis do corpus, só
**163 (14%)** têm alguma legenda na própria seção. Regra de adjacência pura —
a última legenda antes da tabela na ordem do documento, sem olhar a seção — foi
medida também e não cobre mais (13%), além de produzir ligações de até 31
páginas de distância. Fica a da seção.

Duas exigências da forma da legenda pagam sozinhas o que custam, e as duas
saíram do corpus: a pontuação depois do número separa a legenda ("Quadro 5.
Avaliação...") da referência em prosa ("O Quadro 5 apresenta os dados..."), que
está no mesmo pedaço; e o limite de dois dígitos derruba "Quadro 634.", que é o
"Quadro 6" com a chamada de referência "34" colada pela extração.
"""

import re

from geracao_ancorada.fatiamento.pedacos import Pedaco

_LEGENDA = re.compile(
    r"^[ \t]*((?:Quadro|Tabela|Figura)\s*\d{1,2}[ \t]*[.:–-][^\n]*)",
    re.MULTILINE,
)


def legenda(tabela: Pedaco, vizinhos: list[Pedaco]) -> str:
    """A legenda mais próxima antes da tabela, dentro da mesma seção."""
    secao = (tabela.fonte_id, tabela.parte, tabela.secao)
    achada = ""
    for p in vizinhos:
        if p.id == tabela.id:
            break
        if p.tipo != "texto" or (p.fonte_id, p.parte, p.secao) != secao:
            continue
        for m in _LEGENDA.finditer(p.texto):
            achada = m.group(1).strip()
    return achada
