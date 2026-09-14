"""O que entra no índice principal.

Dois cortes. O primeiro é a fonte superada: o manifesto já declara que o PCDT
de diabete de 2026 substitui o de 2024, e nenhum módulo lia esse campo. Com os
dois no mesmo índice, o gerador cita norma revogada com página, que é o modo de
falha que o trabalho existe para medir. O superado volta com
`incluir_superadas=True`, num índice próprio, porque a sonda contrafactual das
insulinas precisa exatamente dele.

O segundo é o apêndice de metodologia. Ele conta como o protocolo foi
elaborado — busca da evidência, consulta pública, equipe, declaração de
conflito de interesse — e não o que fazer com o paciente. A tabela
`pcdt-dm2-2026#ap-ndice-1-metodologia-d/5.5` é a declaração de conflito, com
"Recebeu honorários para apresentação de aulas": casa com consulta sobre
tratamento e não responde nada.

O corte do apêndice é por BLOCO, e não por pedaço, e isso foi medido. Só três
dos seis PCDT trazem o assunto no rótulo ("APÊNDICE 1 – METODOLOGIA DE BUSCA E
AVALIAÇÃO DA LITERATURA"); nos outros três o rótulo é "APÊNDICE 1" pelado e a
marca aparece no título de uma seção ou na primeira linha do texto — 57 pedaços
que a regra de rótulo deixaria entrar. E como a marca costuma estar em uma só
das cinco seções do apêndice, decidir pedaço a pedaço deixaria os outros quatro
quintos dentro. Um apêndice sai inteiro ou fica inteiro.
"""

import re

from geracao_ancorada.fatiamento.pedacos import Pedaco
from geracao_ancorada.fontes.manifesto import Fonte

_APENDICE = re.compile(r"^\s*(AP[ÊE]NDICE|ANEXO)\b", re.IGNORECASE)

_METODOLOGIA = re.compile(
    r"METODOLOGIA\s+DE\s+BUSCA"
    r"|EQUIPE\s+DE\s+ELABORA[ÇC][ÃA]O"
    r"|PARTES\s+INTERESSADAS",
    re.IGNORECASE,
)
"""As três marcas medidas nos onze PDF.

"Equipe" e "elaboração" soltas ficam de fora de propósito: "equipe
multiprofissional" é expressão de clínica e aparece no corpo dos protocolos.
"""

_ABERTURA = 120
"""Quanto do texto conta como primeira linha, em caracteres."""


def superadas(fontes: list[Fonte]) -> frozenset[str]:
    """Os id de fonte que outra fonte do manifesto declara substituir."""
    return frozenset(f.substitui for f in fontes if f.substitui)


def _e_metodologia(p: Pedaco) -> bool:
    return bool(
        _METODOLOGIA.search(p.parte)
        or _METODOLOGIA.search(p.titulo_secao)
        or _METODOLOGIA.search(p.texto[:_ABERTURA])
    )


def indexaveis(
    pedacos: list[Pedaco],
    superadas: frozenset[str] = frozenset(),
    *,
    incluir_superadas: bool = False,
) -> list[Pedaco]:
    """Os pedaços que compõem o índice, na ordem em que chegaram."""
    if not incluir_superadas:
        pedacos = [p for p in pedacos if p.fonte_id not in superadas]

    condenados = {
        (p.fonte_id, p.parte)
        for p in pedacos
        if _APENDICE.match(p.parte) and _e_metodologia(p)
    }
    return [p for p in pedacos if (p.fonte_id, p.parte) not in condenados]
