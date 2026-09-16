"""A fusão das listas dos dois braços por posição, com garantia por braço.

As notas dos dois braços não vivem na mesma escala, então a fusão ignora as
notas e usa só a posição de cada pedaço em cada lista: 1 / (constante +
posição), somado sobre as listas em que o pedaço aparece. Quem falta numa
lista contribui zero por ela, e nunca uma posição imputada.

A fusão por posição premia o consenso, e com a constante 60 o primeiro
colocado de um braço só vale 1/61, menos que qualquer par de posições medianas
nos dois braços. A garantia existe por isso: o primeiro de cada braço entra
sempre na entrega. A decisão inteira, com a aritmética, está em DECISOES.md
(2026-09-15).
"""

CONSTANTE_RRF = 60  # Cormack, Clarke e Buettcher, 2009; declarada, não ajustada
GARANTIDOS_POR_BRACO = 1
PROFUNDIDADE = 50
N_ENTREGUE = 8


def rrf(listas: dict[str, list[str]], k: int = CONSTANTE_RRF) -> list[tuple[str, float]]:
    """Os ids de todas as listas, do maior escore fundido ao menor.

    O empate se resolve pelo id, em ordem alfabética, para que dois pedaços
    quase idênticos saiam sempre na mesma ordem.
    """
    escores: dict[str, float] = {}
    for lista in listas.values():
        for posicao, id_ in enumerate(lista, start=1):
            escores[id_] = escores.get(id_, 0.0) + 1.0 / (k + posicao)
    return sorted(escores.items(), key=lambda par: (-par[1], par[0]))


def aplicar_garantia(
    listas: dict[str, list[str]],
    fundida: list[tuple[str, float]],
    *,
    n: int = N_ENTREGUE,
    garantidos: int = GARANTIDOS_POR_BRACO,
) -> list[tuple[str, float, bool]]:
    """Os `n` entregues, na ordem da fusão, com os garantidos de cada braço dentro.

    A garantia decide quem entra e a fusão decide a ordem: o garantido que a
    fusão pôs em 40º sai no fim da entrega, e não na frente de quem a fusão
    pôs acima dele. A marca no terceiro campo diz que a garantia mudou a
    entrega, isto é, que o pedaço não estaria entre os `n` só pela fusão.
    """
    escore = dict(fundida)
    cabeca: list[str] = []
    for lista in listas.values():
        for id_ in lista[:garantidos]:
            if id_ not in cabeca:
                cabeca.append(id_)

    por_fusao = [id_ for id_, _ in fundida[:n]]
    escolhidos = list(cabeca)
    for id_ in por_fusao:
        if len(escolhidos) >= n:
            break
        if id_ not in escolhidos:
            escolhidos.append(id_)

    entrega = sorted(escolhidos, key=lambda id_: (-escore[id_], id_))[:n]
    return [(id_, escore[id_], id_ not in por_fusao) for id_ in entrega]
