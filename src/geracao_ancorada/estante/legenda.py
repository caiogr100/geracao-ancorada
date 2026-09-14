"""A legenda que ficou no pedaço de texto vizinho da tabela.

Dívida declarada, e o lugar certo do conserto é o fatiador. O PCDT escreve
"Quadro 5. Avaliação clínica e laboratorial para a estimativa de risco
cardiovascular." numa linha e desenha a tabela adiante; o fatiador separa as
duas, e a tabela entra no índice como " | Anamnese | \n- Sexo - Idade -
Tabagismo", sem conter "Quadro 5" nem uma palavra do que ela é.

O alcance foi medido, e é pequeno: das 1.167 tabelas indexáveis do corpus,
**115 (10%)** recebem legenda desta regra. O teto do que seria alcançável sem
sair da seção é 163 (14%), que são as tabelas com alguma legenda na própria
seção, antes ou depois delas. A diferença de 48 são as tabelas cuja única
legenda da seção vem DEPOIS delas.

A regra de adjacência pura, que é a última legenda antes da tabela na ordem do
documento sem olhar a seção, foi medida também. Ela alcança 675 tabelas (58%),
e é esse alcance que a desqualifica: as ligações que ela cria têm mediana de 65
páginas de distância e chegam a 410, então a maior parte do que ela alcança é
legenda de outro assunto. Fica a da seção, com a cobertura menor.

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
