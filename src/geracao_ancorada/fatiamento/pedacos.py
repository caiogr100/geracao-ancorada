"""Montagem dos pedaços: da junta natural até a unidade que vai pra estante.

Cada pedaço tem que passar no teste de ficar em pé sozinho: lido fora do
documento, ele diz de que doença fala, pra quem, e de que ano é. Por isso o
cabeçalho (título do documento, órgão, ano, seção) viaja junto com o texto, e
por isso a página de origem é registrada — é ela que sustenta a citação.

Quando a unidade natural estoura o teto, ela é quebrada nas juntas menores dela
(parágrafo), nunca no escuro.
"""

import re
from dataclasses import dataclass, field

from ..fontes.manifesto import Fonte
from .extracao import Bloco
from .juntas import (
    Junta,
    abre_referencias,
    capitulo,
    detectar,
    inicia_referencias,
    marco,
    subtitulo,
    titulo_sem_marca,
)

TETO = 1800
"""Teto em caracteres. Acima disso a seção é quebrada por parágrafo."""

MINIMO = 120
"""Piso em caracteres. Abaixo disso o pedaço volta a colar no vizinho."""

LISTA_DE_PARTES = 3
"""Marcos na MESMA página a partir dos quais aquilo é a lista de partes.

No manual de tuberculose a página 18 traz sete linhas "anexo i – ...", "anexo
ii – ...", uma atrás da outra: é o sumário dos anexos. Cada uma disparava o
marco, a última grudava, e o corpo do manual saía etiquetado como um anexo que
não é o dele. Um anexo de verdade ocupa a página; três na mesma página são a
lista deles."""

INDICE = 4
"""Títulos numerados seguidos, na mesma página e sem conteúdo entre eles, a
partir dos quais aquilo é um índice e não a espinha do documento.

A Instrução Normativa do Calendário abre listando as 21 vacinas na página 1. Sem
esta regra o índice consome a numeração inteira, o corpo — que recomeça em "1."
— é rejeitado, e cada pedaço herda o rótulo da última linha do índice: foi assim
que o texto da varicela saiu etiquetado como a seção da dT."""


@dataclass(frozen=True)
class Pedaco:
    """Uma unidade de recuperação, com a etiqueta que o roteador vai filtrar."""

    id: str
    fonte_id: str
    titulo_fonte: str
    orgao: str
    ano: int
    areas: list[str] = field(default_factory=list)
    parte: str = ""
    secao: str = ""
    titulo_secao: str = ""
    pagina_inicial: int = 0
    pagina_final: int = 0
    tipo: str = "texto"
    texto: str = ""
    descartavel: bool = False

    @property
    def cabecalho(self) -> str:
        """A moldura que faz o pedaço se sustentar fora do documento."""
        # O PCDT numera a seção e o guia não: "8.1. Metas terapêuticas" contra
        # "PERÍODO DE INCUBAÇÃO". As duas formas saem sem pontuação sobrando,
        # porque esta linha é a citação que o gerador vai imprimir.
        if self.secao and self.titulo_secao:
            secao = f"{self.secao}. {self.titulo_secao}"
        else:
            secao = self.secao or self.titulo_secao or "(sem seção)"
        pagina = (
            f"p. {self.pagina_inicial}"
            if self.pagina_inicial == self.pagina_final
            else f"p. {self.pagina_inicial}-{self.pagina_final}"
        )
        parte = f"{self.parte} — " if self.parte else ""
        return f"{self.titulo_fonte} ({self.orgao}, {self.ano}) — {parte}{secao} — {pagina}"

    @property
    def ancorado(self) -> str:
        """Texto como entra no índice: com a moldura na frente."""
        return f"{self.cabecalho}\n\n{self.texto}"


def _tamanho(blocos: list[Bloco]) -> int:
    return sum(len(b.texto) + 1 for b in blocos) - 1


def _quebrar(blocos: list[Bloco], teto: int, piso: int = MINIMO) -> list[list[Bloco]]:
    """Quebra por parágrafo, e só junta parágrafo enquanto couber no teto.

    Trabalha sobre os blocos, e não sobre o texto já colado, porque cada parte
    precisa saber de que páginas ela veio: é a página que sustenta a citação, e
    uma seção longa que virasse "p. 1-58" não sustenta nada.

    A última parte não pode ficar abaixo do piso por culpa do corte: medido nos
    onze PDF, o piso descartava 125 caudas, e 19 eram a frase que fecha a
    seção. Quando isso acontece, os últimos parágrafos da parte anterior descem
    para a cauda, enquanto ela couber no teto e a anterior não esvaziar. Se não
    há parágrafo que desça ou nenhum a ponha acima do piso, fica como está, e o
    piso decide.
    """
    partes: list[list[Bloco]] = []
    atual: list[Bloco] = []
    tamanho = 0

    for bloco in blocos:
        if atual and tamanho + len(bloco.texto) + 1 > teto:
            partes.append(atual)
            atual, tamanho = [], 0
        atual.append(bloco)
        tamanho += len(bloco.texto) + 1

    if atual:
        partes.append(atual)

    if len(partes) >= 2 and _tamanho(partes[-1]) < piso:
        anterior, cauda = partes[-2], partes[-1]
        for quantos in range(1, len(anterior)):
            candidata = anterior[-quantos:] + cauda
            if _tamanho(candidata) > teto:
                break
            if _tamanho(candidata) >= piso:
                partes[-2], partes[-1] = anterior[:-quantos], candidata
                break
    return partes


def _emitir(
    fonte: Fonte,
    junta: Junta | None,
    acumulado: list[Bloco],
    parte: str,
    suspenso: bool,
) -> list[Pedaco]:
    if not acumulado:
        return []

    secao = junta.numero if junta else ""
    titulo = junta.titulo if junta else ""
    descartavel = suspenso or (junta.descartavel if junta else False)

    pedacos: list[Pedaco] = []
    for tipo in ("texto", "tabela"):
        blocos = [b for b in acumulado if b.tipo == tipo]
        if not blocos:
            continue

        # A tabela é unidade fechada: nunca é fundida com a vizinha nem quebrada.
        grupos = (
            [[b] for b in blocos]
            if tipo == "tabela"
            else [blocos]
        )
        for grupo in grupos:
            for trecho in _quebrar(grupo, TETO):
                pedacos.append(
                    Pedaco(
                        id="",
                        fonte_id=fonte.id,
                        titulo_fonte=fonte.titulo,
                        orgao=fonte.orgao,
                        ano=fonte.ano,
                        areas=list(fonte.areas),
                        parte=parte,
                        secao=secao,
                        titulo_secao=titulo,
                        pagina_inicial=min(b.pagina for b in trecho),
                        pagina_final=max(b.pagina for b in trecho),
                        tipo=tipo,
                        texto="\n".join(b.texto for b in trecho),
                        descartavel=descartavel,
                    )
                )
    return pedacos


def _juntar_curtos(pedacos: list[Pedaco]) -> list[Pedaco]:
    """Funde pedaço curto demais com o vizinho da mesma parte.

    Fragmento solto ("2017;", o cabeçalho de uma coluna) não se sustenta e só
    polui a estante. Aqui ele volta a colar no vizinho enquanto couber no teto.
    """
    saida: list[Pedaco] = []
    for pedaco in pedacos:
        anterior = saida[-1] if saida else None
        cabe = (
            anterior is not None
            and len(anterior.texto) < MINIMO
            and anterior.tipo == pedaco.tipo
            and anterior.descartavel == pedaco.descartavel
            and anterior.parte == pedaco.parte
            # A fronteira de seção é intransponível: colar através dela faz o
            # pedaço sair com o rótulo do vizinho, e a citação passa a apontar
            # para a seção errada. No guia a seção não tem número, então a
            # fronteira mora no título.
            and (anterior.secao, anterior.titulo_secao)
            == (pedaco.secao, pedaco.titulo_secao)
            and len(anterior.texto) + len(pedaco.texto) + 1 <= TETO
        )
        if cabe:
            saida[-1] = Pedaco(
                **{
                    **anterior.__dict__,
                    "texto": f"{anterior.texto}\n{pedaco.texto}",
                    "pagina_final": max(anterior.pagina_final, pedaco.pagina_final),
                }
            )
        else:
            saida.append(pedaco)
    # O piso serve pra varrer fragmento solto, não pra apagar seção curta: se o
    # pedaço ABRE uma seção, ele é todo o conteúdo que ela tem e fica.
    mantidos: list[Pedaco] = []
    for posicao, pedaco in enumerate(saida):
        anterior = saida[posicao - 1] if posicao else None
        abre_secao = anterior is None or (
            (anterior.parte, anterior.secao, anterior.titulo_secao)
            != (pedaco.parte, pedaco.secao, pedaco.titulo_secao)
        )
        if (
            len(pedaco.texto) >= MINIMO
            or pedaco.tipo == "tabela"
            or ((pedaco.secao or pedaco.titulo_secao) and abre_secao)
        ):
            mantidos.append(pedaco)
    return mantidos


def livro_de_capitulo(blocos: list[Bloco]) -> bool:
    """O documento é livro de capítulos, e não de espinha numerada?

    Um livro de capítulos não numera seção, e a regra da sequência — que nos
    PCDT separa o título da referência — não separa o título da LISTA, porque
    uma lista também anda 1, 2, 3. Sem espinha para disputar, a lista ganhava
    sozinha: nos três volumes do Guia, 461 pedaços saíam rotulados com item de
    lista, ficha catalográfica ou trecho de bibliografia, e esse rótulo ia
    impresso na citação.

    O sinal de família é o próprio capítulo. Dos onze documentos do corpus, só
    os três volumes do Guia têm capítulo; o manual de tuberculose e os sete
    PCDT não têm, e a numeração deles segue valendo.
    """
    texto = [b.texto for b in blocos if b.tipo == "texto"]
    return any(
        capitulo(linha, texto[i + 1] if i + 1 < len(texto) else "", texto[i - 1] if i else "")
        for i, linha in enumerate(texto)
    )


def fatiar(fonte: Fonte, blocos: list[Bloco]) -> list[Pedaco]:
    """Agrupa os blocos por seção e devolve os pedaços já identificados."""
    livro = livro_de_capitulo(blocos)
    pedacos: list[Pedaco] = []
    junta: Junta | None = None
    nivel: tuple[int, ...] | None = None
    acumulado: list[Bloco] = []
    parte = ""
    suspenso = False
    seguidas = 0
    pagina_junta = None
    indice = False
    marcos_na_pagina = 0
    pagina_marco = None

    for posicao, bloco in enumerate(blocos):
        if bloco.tipo == "texto":
            # O capítulo de guia só se confirma pela linha seguinte, então
            # é preciso espiar adiante: o título sozinho é indistinguível de
            # qualquer outra linha em caixa alta. E espiar atrás, porque o
            # título que não coube numa linha continua na de baixo.
            adiante = blocos[posicao + 1].texto if posicao + 1 < len(blocos) else ""
            atras = blocos[posicao - 1].texto if posicao else ""
            nome = capitulo(bloco.texto, adiante, atras) or marco(bloco.texto)
            # Dentro da bibliografia, a linha de citação também começa com
            # "anexo" ("Portaria de Consolidação nº 2, anexo XXV. Disponível
            # em:", "Anexo V – Sistema Nacional de Vigilância Epidemiológica")
            # e reabria a estrutura no meio das referências, que saíam úteis.
            # O título de parte vem em caixa alta; a citação, não.
            if nome is not None and suspenso and nome != nome.upper():
                nome = None
            if nome is not None:
                # Título remontado: a primeira metade já entrou no acumulado e
                # sairia como conteúdo do capítulo anterior.
                if acumulado and nome.startswith(atras.strip()) and nome != atras.strip():
                    acumulado.pop()
                if bloco.pagina != pagina_marco:
                    pagina_marco, marcos_na_pagina = bloco.pagina, 0
                marcos_na_pagina += 1
                pedacos.extend(_emitir(fonte, junta, acumulado, parte, suspenso))
                junta, nivel, acumulado = None, None, []
                # Passou do teto na mesma página: aquilo é a lista das partes, e
                # nenhuma delas abre de verdade aqui. Fica sem parte até a
                # próxima, que virá numa página só dela.
                parte = "" if marcos_na_pagina >= LISTA_DE_PARTES else nome
                suspenso = False
                seguidas, indice = 0, False
                continue

        # No livro de capítulos a seção é o subtítulo marcado com a chave, e a
        # bibliografia que fecha cada capítulo é suspensa até o capítulo
        # seguinte — é de lá que saía o rótulo de parte "Anexo V – Sistema
        # Nacional de Vigilância Epidemiológica", que é linha de citação ABNT.
        if livro:
            if bloco.tipo == "texto":
                nome = subtitulo(bloco.texto)
                if nome is not None:
                    # O subtítulo também FECHA a bibliografia. Fechar só no
                    # capítulo seguinte não bastava: as sete primeiras seções
                    # do volume 1 são gerais e não têm CID, então a suspensão
                    # de uma atravessava a seguinte inteira. Referência ABNT
                    # não vem marcada nem em caixa alta, e por isso não reabre
                    # por engano.
                    pedacos.extend(_emitir(fonte, junta, acumulado, parte, suspenso))
                    junta, acumulado, suspenso = Junta(numero="", titulo=nome, nivel=()), [], False
                    continue
                # O cabeçalho da bibliografia vem antes do título sem marca,
                # porque "BIBLIOGRAFIA" também é caixa alta sem ponto: o
                # capítulo de influenza tem REFERÊNCIAS e, logo depois,
                # BIBLIOGRAFIA, e o segundo cabeçalho era tomado por título e
                # reabria o capítulo no meio das citações.
                if abre_referencias(bloco.texto):
                    pedacos.extend(_emitir(fonte, junta, acumulado, parte, suspenso))
                    junta, acumulado = None, []
                    suspenso = True
                    continue
                # O título pelado da seção geral só fecha a bibliografia; ele
                # não vira rótulo, porque no capítulo de doença a mesma forma
                # é o nível de cima do subtítulo ("CARACTERÍSTICAS GERAIS"
                # acima de "} DESCRIÇÃO") e sairia por cima do rótulo fino.
                if suspenso and titulo_sem_marca(bloco.texto):
                    # Fecha a bibliografia ANTES de virar a chave: o que está
                    # acumulado ainda é citação e tem que sair descartável.
                    pedacos.extend(_emitir(fonte, junta, acumulado, parte, suspenso))
                    junta, acumulado, suspenso = None, [bloco], False
                    continue
            acumulado.append(bloco)
            continue

        nova = (
            detectar(bloco.texto, nivel)
            if bloco.tipo == "texto" and not suspenso
            else None
        )
        if nova is None:
            # O cabeçalho de referências suspende por si, mesmo sem ter sido
            # aceito como seção: a lista numerada que vem depois não pode virar
            # a espinha do documento.
            if bloco.tipo == "texto" and not suspenso and inicia_referencias(bloco.texto):
                pedacos.extend(_emitir(fonte, junta, acumulado, parte, suspenso))
                junta, nivel, acumulado = None, None, []
                suspenso = True
                continue
            if indice:
                # O índice acabou. A numeração dele não é a espinha do corpo,
                # então zera pra que o corpo possa recomeçar em "1.".
                pedacos.extend(_emitir(fonte, junta, acumulado, parte, suspenso))
                junta, nivel, acumulado = None, None, []
                seguidas, indice = 0, False
            acumulado.append(bloco)
            continue

        # Título que chega sem conteúdo desde o anterior e na mesma página: ou é
        # seção que só tem subseções, ou é linha de índice. A contagem separa as
        # duas — uma seção assim aparece isolada, o índice vem em fila.
        vazia = not acumulado and junta is not None and pagina_junta == bloco.pagina
        seguidas = seguidas + 1 if vazia else 1
        indice = indice or seguidas >= INDICE

        pedacos.extend(_emitir(fonte, junta, acumulado, parte, suspenso))
        junta, nivel, acumulado = nova, nova.nivel, []
        pagina_junta = bloco.pagina
        # Entrou em REFERÊNCIAS: dali em diante nenhum número é seção, porque a
        # própria lista é numerada e continuaria a sequência por acidente.
        if nova.descartavel:
            suspenso = True

    pedacos.extend(_emitir(fonte, junta, acumulado, parte, suspenso))
    pedacos = _juntar_curtos(pedacos)

    numerados: list[Pedaco] = []
    contagem: dict[str, int] = {}
    for pedaco in pedacos:
        parte = re.sub(r"[^A-Za-z0-9]+", "-", pedaco.parte).strip("-").lower()[:24]
        secao = pedaco.secao or re.sub(
            r"[^A-Za-z0-9]+", "-", pedaco.titulo_secao
        ).strip("-").lower()[:24]
        chave = f"{pedaco.fonte_id}#{parte + '/' if parte else ''}{secao or 'preambulo'}"
        contagem[chave] = contagem.get(chave, 0) + 1
        numerados.append(
            Pedaco(**{**pedaco.__dict__, "id": f"{chave}.{contagem[chave]}"})
        )
    return numerados
