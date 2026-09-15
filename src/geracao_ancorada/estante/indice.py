"""O registro do índice, a assinatura e o cache de vetores.

Dois destinos com papéis opostos. O pesado, com o texto de cada pedaço e a
matriz de vetores, mora em `fontes/cache/estante/`, que o `.gitignore` já
cobre pela regra dos PDF. O leve, com uma linha de etiquetas por pedaço e a
assinatura do que foi indexado, mora em `indice/` e é versionado: é o que os
orientadores leem, e não contém uma linha de texto de fonte.

O índice é sempre reconstruído inteiro. O que poupa o Ollama é o cache
endereçado pelo conteúdo, com chave `sha256(modelo + digest do modelo + texto
que entra no encoder)`; reindexação parcial no lugar não existe, porque a classe de defeito
"matriz e registro dessincronizados" deixa de existir em vez de ser testada.

A posição na matriz é o identificador do pedaço em todo o caminho. Por isso o
registro só tem os pedaços que o servidor ACEITOU, alinhados linha a linha com
`vetores.npy`; os recusados ficam na assinatura, com o hash do texto e não só
com o id, porque o id é posicional (`preambulo.41` é o 41º daquele grupo) e
muda de número quando o fatiador muda. A comparação que avisa se a lista de
recusados cresceu se faz pelo hash.
"""

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from geracao_ancorada.estante import bm25, lexico, vetores
from geracao_ancorada.estante.composicao import indexaveis, superadas
from geracao_ancorada.estante.legenda import legenda
from geracao_ancorada.estante.texto import texto_indexado_denso, texto_indexado_lexico
from geracao_ancorada.fatiamento.extracao import Bloco, extrair
from geracao_ancorada.fatiamento.pedacos import Pedaco, fatiar
from geracao_ancorada.fontes.manifesto import Fonte

RAIZ = Path(__file__).resolve().parents[3]
CACHE_PDF = RAIZ / "fontes" / "cache"
DESTINO = CACHE_PDF / "estante"
REGISTRO = RAIZ / "indice"

_PEDACOS = "pedacos.jsonl"
_VETORES = "vetores.npy"
_CACHE = "vetores-cache.jsonl"
_REGISTRO = "registro.json"
_ASSINATURA = "assinatura.json"

_CAMPOS_PUBLICOS = (
    "id", "fonte_id", "parte", "secao", "titulo_secao", "pagina_inicial",
    "pagina_final", "tipo", "areas", "ano",
)


class IndiceDesalinhado(Exception):
    """O que está em disco não é um índice só: matriz, registro e assinatura divergem."""


class IndiceDeOutroModelo(Exception):
    """O modelo em execução não é o que construiu o índice."""


def digerir(texto: str) -> str:
    """O hash do texto, curto o bastante para caber numa linha de registro."""
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


def _sha256(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


@dataclass(frozen=True)
class Assinatura:
    """O que foi indexado, com o quê, e o que ficou de fora."""

    n_pedacos: int
    fontes_indexadas: list[str]
    fontes_de_fora: dict[str, str]
    incluir_superadas: bool
    recusados: list[dict]
    sha256_pedacos: str
    sha256_ids: str
    sha256_vetores: str
    modelo: str
    digest_modelo: str
    dimensao: int
    versao_ollama: str
    k1: float
    b: float
    versao_tokenizador: str
    bm25: dict
    commit_fatiador: str
    data: str


@dataclass(frozen=True)
class Estante:
    """O índice carregado: os pedaços, a matriz densa e o índice léxico."""

    pedacos: list[Pedaco]
    legendas: list[str]
    matriz: np.ndarray
    bm25: bm25.BM25
    assinatura: Assinatura


# --- o que se injeta ---------------------------------------------------------


def _blocos_de(fonte: Fonte) -> list[Bloco] | None:
    caminho = CACHE_PDF / f"{fonte.id}.pdf"
    return extrair(caminho) if caminho.exists() else None


def _modelo_em_execucao(modelo: str = vetores.MODELO) -> tuple[str, str, str]:
    """Nome, digest e versão do servidor, lidos do Ollama que está no ar."""
    import httpx

    modelos = httpx.get("http://localhost:11434/api/tags", timeout=10.0).json()["models"]
    digest = next(
        (m["digest"] for m in modelos if m["name"].split(":")[0] == modelo.split(":")[0]),
        "",
    )
    versao = httpx.get("http://localhost:11434/api/version", timeout=10.0).json()["version"]
    return modelo, digest, versao


def _commit() -> str:
    """O commit do código que indexou, com "-dirty" se a árvore tinha mudança
    sem commit. A primeira assinatura do corpus dizia d0ecd57 e foi produzida
    por código que esse commit não continha."""
    try:
        saida = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=7"],
            cwd=RAIZ, capture_output=True, text=True, check=True,
        )
        return saida.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconhecido"


def _agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --- indexar ------------------------------------------------------------------


def _chave(modelo: str, digest: str, texto: str) -> str:
    """A chave leva o digest, e não só o nome: um `ollama pull` que troque o
    modelo mantém o nome, e o cache reaproveitaria todos os vetores velhos."""
    return _sha256(f"{modelo}\x00{digest}\x00{texto}".encode("utf-8"))


def _ler_cache(caminho: Path) -> dict[str, list[float]]:
    if not caminho.exists():
        return {}
    cache = {}
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if linha:
            entrada = json.loads(linha)
            cache[entrada["chave"]] = entrada["vetor"]
    return cache


def _gravar_cache(caminho: Path, cache: dict[str, list[float]]) -> None:
    linhas = [
        json.dumps({"chave": chave, "vetor": cache[chave]}, separators=(",", ":"))
        for chave in sorted(cache)
    ]
    caminho.write_text("\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8")


def _legendas(pedacos: list[Pedaco]) -> list[str]:
    """A legenda de cada tabela, procurada só entre os pedaços da mesma fonte."""
    por_fonte: dict[str, list[Pedaco]] = {}
    for p in pedacos:
        por_fonte.setdefault(p.fonte_id, []).append(p)
    return [
        legenda(p, por_fonte[p.fonte_id]) if p.tipo == "tabela" else ""
        for p in pedacos
    ]


def indexar(
    fontes: list[Fonte],
    *,
    incluir_superadas: bool = False,
    modelo: str = vetores.MODELO,
    destino: Path = DESTINO,
    registro: Path = REGISTRO,
    blocos_de=_blocos_de,
    chamar=vetores._chamar_ollama,
    modelo_em_execucao=_modelo_em_execucao,
    commit=_commit,
    agora=_agora,
) -> Assinatura:
    """Reconstrói o índice inteiro a partir do manifesto e grava os dois destinos."""
    # Lido antes de qualquer escrita: o registro é versionado, e reescrevê-lo
    # é o que fazia o `git describe --dirty` acusar a própria saída do índice.
    commit_lido = commit()
    sup = superadas(fontes)
    de_fora: dict[str, str] = {}
    uteis: list[Pedaco] = []
    for fonte in fontes:
        if fonte.id in sup and not incluir_superadas:
            de_fora[fonte.id] = "superada"
            continue
        blocos = blocos_de(fonte)
        if blocos is None:
            de_fora[fonte.id] = "sem_pdf"
            continue
        pedacos = [p for p in fatiar(fonte, blocos) if not p.descartavel]
        if not pedacos:
            de_fora[fonte.id] = "espinha_ausente"
            continue
        uteis.extend(pedacos)

    pedacos = indexaveis(uteis, sup, incluir_superadas=incluir_superadas)
    legendas = _legendas(pedacos)
    densos = [texto_indexado_denso(p, leg) for p, leg in zip(pedacos, legendas)]

    # O cache poupa o servidor; quem não está nele é vetorizado agora, e quem
    # o servidor recusar fica de fora da matriz e dentro da assinatura.
    nome, digest, versao = modelo_em_execucao()
    if not digest:
        raise IndiceDeOutroModelo(
            f"o servidor não tem o modelo {modelo}, e um índice sem digest carregaria sempre"
        )
    destino.mkdir(parents=True, exist_ok=True)
    cache = _ler_cache(destino / _CACHE)
    chaves = [_chave(modelo, digest, t) for t in densos]
    faltam = [i for i, chave in enumerate(chaves) if chave not in cache]
    resultado = vetores.vetorizar_corpus(
        [densos[i] for i in faltam], modelo=modelo, chamar=chamar
    )
    for linha, posicao in enumerate(resultado.aceitos):
        cache[chaves[faltam[posicao]]] = resultado.matriz[linha].tolist()
    recusados_posicoes = {faltam[posicao] for posicao in resultado.recusados}
    _gravar_cache(destino / _CACHE, cache)

    aceitos = [i for i in range(len(pedacos)) if i not in recusados_posicoes]
    matriz = (
        np.asarray([cache[chaves[i]] for i in aceitos], dtype=np.float32)
        if aceitos
        else np.zeros((0, vetores.DIMENSAO), dtype=np.float32)
    )
    recusados = [
        {
            "id": pedacos[i].id,
            "sha256_texto": digerir(pedacos[i].texto),
            "n_caracteres": len(densos[i]),
        }
        for i in sorted(recusados_posicoes)
    ]
    pedacos = [pedacos[i] for i in aceitos]
    legendas = [legendas[i] for i in aceitos]

    # O destino pesado, com texto.
    linhas = [
        json.dumps({**asdict(p), "legenda": leg}, ensure_ascii=False, separators=(",", ":"))
        for p, leg in zip(pedacos, legendas)
    ]
    pesado = ("\n".join(linhas) + "\n").encode("utf-8")
    (destino / _PEDACOS).write_bytes(pesado)
    np.save(destino / _VETORES, matriz)

    # O destino leve, sem texto.
    registro.mkdir(parents=True, exist_ok=True)
    publicos = [
        {
            **{campo: getattr(p, campo) for campo in _CAMPOS_PUBLICOS},
            "sha256_texto": digerir(p.texto),
            "n_caracteres": len(p.texto),
        }
        for p in pedacos
    ]
    (registro / _REGISTRO).write_text(
        json.dumps({"pedacos": publicos}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    indice_lexico = bm25.construir(
        [lexico.tokenizar(texto_indexado_lexico(p, leg)) for p, leg in zip(pedacos, legendas)]
    )
    assinatura = Assinatura(
        n_pedacos=len(pedacos),
        fontes_indexadas=sorted({p.fonte_id for p in pedacos}, key=[p.fonte_id for p in pedacos].index),
        fontes_de_fora=de_fora,
        incluir_superadas=incluir_superadas,
        recusados=recusados,
        sha256_pedacos=_sha256(pesado),
        sha256_ids=_sha256("\n".join(p.id for p in pedacos).encode("utf-8")),
        sha256_vetores=_sha256((destino / _VETORES).read_bytes()),
        modelo=modelo,
        digest_modelo=digest,
        dimensao=vetores.DIMENSAO,
        versao_ollama=versao,
        k1=bm25.K1,
        b=bm25.B,
        versao_tokenizador=lexico.VERSAO,
        bm25={
            "n_documentos": indice_lexico.n_documentos,
            "comprimento_medio": indice_lexico.comprimento_medio,
            "vocabulario": len(indice_lexico.postings),
        },
        commit_fatiador=commit_lido,
        data=agora(),
    )
    (registro / _ASSINATURA).write_text(
        json.dumps(asdict(assinatura), ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return assinatura


# --- carregar -----------------------------------------------------------------


def carregar(
    *,
    destino: Path = DESTINO,
    registro: Path = REGISTRO,
    modelo_em_execucao=_modelo_em_execucao,
) -> Estante:
    """Carrega o índice, ou se recusa quando ele não é um índice só.

    A assinatura grava três hashes, e os três são conferidos aqui, cada um
    contra o arquivo que ele resume: a lista de ids contra o registro, os
    bytes de `pedacos.jsonl` e os de `vetores.npy` contra o destino pesado.
    Os dois últimos são o que pega "regenerei o pesado e esqueci o leve", em
    que os ids continuam iguais e só o conteúdo mudou. A versão do tokenizador
    também: o BM25 é reconstruído aqui com o tokenizador de agora, e se ele
    não é o que assinou o índice, a estatística gravada na assinatura mente.
    """
    assinatura = Assinatura(
        **json.loads((registro / _ASSINATURA).read_text(encoding="utf-8"))
    )
    _, digest, _ = modelo_em_execucao()
    if digest != assinatura.digest_modelo:
        raise IndiceDeOutroModelo(
            f"o índice foi construído com {assinatura.modelo} ({assinatura.digest_modelo[:12]}) "
            f"e o servidor está com {digest[:12]}"
        )
    if assinatura.versao_tokenizador != lexico.VERSAO:
        raise IndiceDesalinhado(
            f"o índice foi tokenizado pela versão {assinatura.versao_tokenizador} "
            f"e o código está na {lexico.VERSAO}; reindexe"
        )

    publicos = json.loads((registro / _REGISTRO).read_text(encoding="utf-8"))["pedacos"]
    ids_registro = [linha["id"] for linha in publicos]
    if _sha256("\n".join(ids_registro).encode("utf-8")) != assinatura.sha256_ids:
        raise IndiceDesalinhado("a lista de ids do registro não é a da assinatura")

    pesado = (destino / _PEDACOS).read_bytes()
    if _sha256(pesado) != assinatura.sha256_pedacos:
        raise IndiceDesalinhado("os pedaços em disco não são os que a assinatura gravou")
    pedacos: list[Pedaco] = []
    legendas: list[str] = []
    nomes = {campo.name for campo in fields(Pedaco)}
    for linha in pesado.decode("utf-8").splitlines():
        if not linha:
            continue
        dados = json.loads(linha)
        legendas.append(dados.pop("legenda", ""))
        pedacos.append(Pedaco(**{k: v for k, v in dados.items() if k in nomes}))
    if [p.id for p in pedacos] != ids_registro:
        raise IndiceDesalinhado("os pedaços em disco não são os do registro")

    if _sha256((destino / _VETORES).read_bytes()) != assinatura.sha256_vetores:
        raise IndiceDesalinhado("a matriz em disco não é a que a assinatura gravou")
    matriz = np.load(destino / _VETORES)
    if matriz.shape[0] != len(pedacos):
        raise IndiceDesalinhado(
            f"a matriz tem {matriz.shape[0]} linhas e o registro tem {len(pedacos)} pedaços"
        )

    indice_lexico = bm25.construir(
        [lexico.tokenizar(texto_indexado_lexico(p, leg)) for p, leg in zip(pedacos, legendas)]
    )
    return Estante(
        pedacos=pedacos,
        legendas=legendas,
        matriz=matriz,
        bm25=indice_lexico,
        assinatura=assinatura,
    )
