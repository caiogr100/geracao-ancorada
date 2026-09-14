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


def _pedir(textos, modelo, keep_alive, chamar):
    corpo = {
        "model": modelo,
        "input": list(textos),
        "truncate": False,
        "keep_alive": keep_alive,
    }
    status, dados = chamar(corpo)
    if status == 400:
        erro = dados.get("error", "") if isinstance(dados, dict) else ""
        raise TextoLongoDemais(f"{_culpado(textos, modelo, keep_alive, chamar)}: {erro}")
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
    for posicao, parte in enumerate(partes):
        ultimo = posicao == len(partes) - 1
        bruto = _pedir(
            parte, modelo, DESCARREGAR if ultimo else RESIDENTE, chamar
        )
        matrizes.append(_normalizar(bruto))
    return np.vstack(matrizes)


def vetorizar_consulta(
    texto: str, *, modelo: str = MODELO, chamar=_chamar_ollama
) -> np.ndarray:
    """O vetor da pergunta, sem descarregar o modelo depois."""
    bruto = _pedir([texto], modelo, RESIDENTE, chamar)
    return _normalizar(bruto)[0]
