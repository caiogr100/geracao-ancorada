"""O braço denso: os pedaços virando vetor pelo Ollama.

O risco aqui não é errar a conta, é acertá-la sobre um texto que chegou pela
metade. Com `truncate: true` o Ollama corta o que passa do teto de contexto e
devolve 200 com um vetor de aparência perfeitamente normal, e a tabela cortada
no meio entra no índice, pontua na busca e não contém a linha da dose. Por isso
todo pedido sai com `truncate: false`, e o 400 vira `TextoLongoDemais` dizendo
qual texto estourou.

O teto foi medido nesta máquina, e não é o que o desenho supunha. Pedir
`num_ctx` maior não levanta nada: o Ollama obedece o valor para baixo e satura
em torno de 2.048 tokens, com 4.096, 8.192 e 16.384 dando o mesmo resultado. E
o teto em caracteres depende da densidade do texto, cerca de 8.900 em prosa
corrida contra 3.300 numa tabela cheia de número e de barra vertical. Por isso
não existe aqui a conferência de comprimento em caracteres que o desenho pedia:
ela seria frouxa demais para a tabela e apertada demais para a prosa. Quem
sabe o teto é o servidor, e a resposta dele é o 400.

A normalização é explícita mesmo com o Ollama já devolvendo norma 1, porque é
ela que faz o produto interno do índice ser cosseno. Se um modelo trocado
devolver vetor cru, a conta continua certa em vez de sair errada em silêncio.

As duas funções são iguais hoje de propósito. O BGE-M3 é simétrico e não usa
prefixo de instrução; o `query:` e o `passage:` são do E5, que é o candidato de
troca. Mantê-las separadas custa uma linha agora e evita que a troca compile,
rode e devolva recall pior sem erro nenhum.
"""

from dataclasses import dataclass

import numpy as np

MODELO = "bge-m3"
DIMENSAO = 1024
LOTE = 16
RESIDENTE = "5m"
DESCARREGAR = "0"

_URL = "http://localhost:11434/api/embed"


class TextoLongoDemais(Exception):
    """O Ollama recusou o texto por estourar o contexto do modelo."""


def _chamar_ollama(corpo: dict) -> tuple[int, dict]:
    import httpx

    resposta = httpx.post(_URL, json=corpo, timeout=600.0)
    return resposta.status_code, resposta.json()


def _pedir(textos, modelo, keep_alive, chamar, *, isolar=True):
    """Um pedido ao servidor. Com `isolar`, o 400 de um lote vira uma mensagem
    que nomeia o texto culpado, ao custo de um pedido por texto do lote; quem
    vai isolar por conta própria, como a indexação, passa `isolar=False`."""
    corpo = {
        "model": modelo,
        "input": list(textos),
        "truncate": False,
        "keep_alive": keep_alive,
    }
    status, dados = chamar(corpo)
    if status == 400:
        erro = dados.get("error", "") if isinstance(dados, dict) else ""
        quem = (
            _culpado(textos, modelo, keep_alive, chamar)
            if isolar
            else f"o Ollama recusou um lote de {len(textos)}"
        )
        raise TextoLongoDemais(f"{quem}: {erro}")
    if status != 200:
        raise RuntimeError(f"o Ollama devolveu {status}: {dados}")
    return dados["embeddings"]


def _culpado(textos, modelo, keep_alive, chamar) -> str:
    """Qual texto do lote o Ollama recusou.

    O 400 vem do lote inteiro, e sem isolar a mensagem acusaria o primeiro
    texto dele, que quase sempre é um pedaço inocente. A repetição um a um só
    acontece no caminho do erro, então ela não custa nada no caminho normal.
    """
    if len(textos) == 1:
        return f"o Ollama recusou o texto {textos[0][:60].replace(chr(10), ' ')!r}"

    recusados = []
    for texto in textos:
        corpo = {
            "model": modelo,
            "input": [texto],
            "truncate": False,
            "keep_alive": keep_alive,
        }
        if chamar(corpo)[0] == 400:
            recusados.append(repr(texto[:60].replace("\n", " ")))
    if not recusados:
        return "o Ollama recusou o lote, mas aceita cada texto dele sozinho"
    return "o Ollama recusou " + ", ".join(recusados)


def _descarregar(modelo, chamar) -> None:
    """Solta a memória de vídeo que o modelo de embedding está segurando.

    O gerador entra logo depois da indexação e disputa a mesma placa. Amarrar
    isso ao último lote foi a primeira tentativa e não sobrevive ao caminho do
    erro, onde o último lote pode ser justamente o que foi recusado; por isso é
    um passo próprio. A entrada vazia foi conferida contra o servidor, que a
    aceita e devolve 200.

    Roda no `finally` das duas portas, então não pode levantar nada: se o
    servidor caiu, o modelo já não está residente, e se respondeu outra coisa,
    a exceção que interessa é a que já está em voo.
    """
    corpo = {"model": modelo, "input": [""], "truncate": False, "keep_alive": DESCARREGAR}
    try:
        chamar(corpo)
    except Exception:
        pass


def _normalizar(bruto: list[list[float]]) -> np.ndarray:
    matriz = np.asarray(bruto, dtype=np.float32)
    if matriz.shape[1] != DIMENSAO:
        raise ValueError(
            f"o modelo devolveu {matriz.shape[1]} dimensões e o índice usa {DIMENSAO}"
        )
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    if not np.all(normas > 0):
        raise ValueError("o modelo devolveu vetor nulo, que não tem direção")
    return (matriz / normas).astype(np.float32)


def vetorizar_documentos(
    textos: list[str],
    *,
    modelo: str = MODELO,
    lote: int = LOTE,
    chamar=_chamar_ollama,
) -> np.ndarray:
    """A matriz (n, 1024) dos pedaços, na ordem em que entraram."""
    if not textos:
        return np.zeros((0, DIMENSAO), dtype=np.float32)

    partes = [textos[i : i + lote] for i in range(0, len(textos), lote)]
    matrizes = []
    try:
        for parte in partes:
            matrizes.append(_normalizar(_pedir(parte, modelo, RESIDENTE, chamar)))
    finally:
        _descarregar(modelo, chamar)
    return np.vstack(matrizes)


@dataclass(frozen=True)
class Vetorizacao:
    """O resultado de vetorizar um corpus inteiro, com quem ficou de fora.

    A posição na matriz identifica o pedaço em todo o resto do caminho, então
    quem indexa precisa descartar do registro exatamente as posições listadas
    em `recusados`, na mesma passada.
    """

    matriz: np.ndarray
    aceitos: list[int]
    recusados: list[int]


def vetorizar_corpus(
    textos: list[str],
    *,
    modelo: str = MODELO,
    lote: int = LOTE,
    chamar=_chamar_ollama,
) -> Vetorizacao:
    """Vetoriza o corpus anotando quem o servidor recusou, sem abortar.

    A diferença para `vetorizar_documentos` é de política, e ela é deliberada.
    Uma consulta recusada é um defeito e tem que estourar na hora. Um pedaço
    recusado na indexação é um fato do corpus: a indexação segue sem ele e o
    registra, para que a lista de ausentes seja conferível depois em vez de
    depender de alguém ter lido a saída do terminal.
    """
    aceitos: list[int] = []
    recusados: list[int] = []
    matrizes = []
    partes = [
        list(range(i, min(i + lote, len(textos)))) for i in range(0, len(textos), lote)
    ]
    try:
        for indices in partes:
            try:
                bruto = _pedir(
                    [textos[i] for i in indices], modelo, RESIDENTE, chamar, isolar=False
                )
            except TextoLongoDemais:
                # O 400 condena o lote, e não o texto. Aqui cada um é tentado
                # sozinho, e só quem for recusado de novo fica de fora. O
                # isolamento é este, uma vez só: com o de `_pedir` ligado, o
                # lote de 16 com um recusado custava 34 pedidos.
                for i in indices:
                    try:
                        um = _pedir([textos[i]], modelo, RESIDENTE, chamar, isolar=False)
                    except TextoLongoDemais:
                        recusados.append(i)
                    else:
                        aceitos.append(i)
                        matrizes.append(_normalizar(um))
                continue
            aceitos.extend(indices)
            matrizes.append(_normalizar(bruto))
    finally:
        # No finally, e não depois do laço: o erro que não é 400 também sai
        # daqui, e o gerador não pode entrar em cima de um modelo residente.
        _descarregar(modelo, chamar)

    vazia = np.zeros((0, DIMENSAO), dtype=np.float32)
    return Vetorizacao(
        matriz=np.vstack(matrizes) if matrizes else vazia,
        aceitos=aceitos,
        recusados=recusados,
    )


def vetorizar_consulta(
    texto: str, *, modelo: str = MODELO, chamar=_chamar_ollama
) -> np.ndarray:
    """O vetor da pergunta, sem descarregar o modelo depois."""
    bruto = _pedir([texto], modelo, RESIDENTE, chamar)
    return _normalizar(bruto)[0]
