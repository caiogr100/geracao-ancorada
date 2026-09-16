"""O registro do índice, a assinatura e o cache de vetores.

O índice é reconstruído inteiro toda vez; o que poupa o Ollama é o cache
endereçado pelo conteúdo. A posição na matriz é o identificador do pedaço em
todo o caminho, por isso o registro só tem os pedaços que o servidor aceitou,
alinhados linha a linha com `vetores.npy`, e os recusados ficam na assinatura.

Nada aqui toca em PDF nem em GPU: a extração, o Ollama, o commit e o relógio
entram por injeção, como em `test_vetores.py`.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from geracao_ancorada.estante import indice
from geracao_ancorada.estante.indice import (
    IndiceDeOutroModelo,
    IndiceDesalinhado,
    carregar,
    digerir,
    indexar,
)
from geracao_ancorada.estante.vetores import DIMENSAO
from geracao_ancorada.fatiamento.extracao import Bloco
from geracao_ancorada.fontes.manifesto import Fonte

RAIZ = Path(__file__).resolve().parents[1]


def fonte(id, substitui=None):
    return Fonte(
        id=id,
        titulo=f"Protocolo {id}",
        orgao="MS/CONITEC",
        ano=2026,
        url=f"https://exemplo.invalido/{id}.pdf",
        base_legal="Portaria de exemplo",
        redistribuivel=True,
        areas=["clinica-medica"],
        substitui=substitui,
    )


def texto(pagina, corpo):
    return Bloco(pagina=pagina, tipo="texto", texto=corpo)


def documento(marca):
    """Três seções com conteúdo, o suficiente para três pedaços."""
    return [
        texto(1, "1. INTRODUÇÃO"),
        texto(1, f"O diabete melito tipo 2 {marca} " + "a" * 200),
        texto(2, "2. DIAGNÓSTICO"),
        texto(2, f"O diagnóstico laboratorial {marca} " + "b" * 200),
        texto(3, "3. TRATAMENTO"),
        texto(3, f"Insulina NPH {marca} " + "c" * 200),
    ]


def ollama_falso(recusar=()):
    """Vetoriza pela direção do texto e recusa quem contiver uma das marcas."""
    chamadas = []

    def chamar(corpo):
        chamadas.append(corpo)
        if any(marca in t for t in corpo["input"] for marca in recusar):
            return 400, {"error": "the input length exceeds the context length"}
        vetores = []
        for t in corpo["input"]:
            v = [0.0] * DIMENSAO
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % DIMENSAO] = 1.0
            vetores.append(v)
        return 200, {"embeddings": vetores}

    chamar.chamadas = chamadas
    return chamar


def ambiente(tmp_path, blocos, chamar=None, digest="digest-falso"):
    """Tudo o que `indexar` e `carregar` precisam, apontando para tmp_path."""
    return dict(
        destino=tmp_path / "estante",
        registro=tmp_path / "indice",
        blocos_de=lambda f: blocos.get(f.id),
        chamar=chamar or ollama_falso(),
        modelo_em_execucao=lambda: ("bge-m3", digest, "0.34.0"),
        commit=lambda: "abc1234",
        agora=lambda: "2026-09-14T00:00:00",
    )


# --- o que se grava ---------------------------------------------------------


def test_indexar_alinha_registro_e_matriz(tmp_path):
    fontes = [fonte("pcdt-a"), fonte("pcdt-b")]
    amb = ambiente(tmp_path, {"pcdt-a": documento("alfa"), "pcdt-b": documento("beta")})

    assinatura = indexar(fontes, **amb)
    estante = carregar(destino=amb["destino"], registro=amb["registro"],
                       modelo_em_execucao=amb["modelo_em_execucao"])

    assert assinatura.n_pedacos == 6
    assert estante.matriz.shape == (6, DIMENSAO)
    assert len(estante.pedacos) == 6
    assert [p.fonte_id for p in estante.pedacos] == ["pcdt-a"] * 3 + ["pcdt-b"] * 3


def test_indexar_registra_as_fontes_que_ficaram_de_fora(tmp_path):
    """Superada e sem PDF, cada uma com o seu motivo.

    Sem isso, recall baixo em preventiva seria caçado no BM25 quando o
    documento nunca entrou no índice.
    """
    fontes = [fonte("pcdt-2026", substitui="pcdt-2024"), fonte("pcdt-2024"), fonte("guia-v9")]
    amb = ambiente(tmp_path, {"pcdt-2026": documento("nova"), "pcdt-2024": documento("velha")})

    assinatura = indexar(fontes, **amb)

    assert assinatura.fontes_indexadas == ["pcdt-2026"]
    assert assinatura.fontes_de_fora == {"pcdt-2024": "superada", "guia-v9": "sem_pdf"}


def test_superada_volta_quando_pedida(tmp_path):
    fontes = [fonte("pcdt-2026", substitui="pcdt-2024"), fonte("pcdt-2024")]
    amb = ambiente(tmp_path, {"pcdt-2026": documento("nova"), "pcdt-2024": documento("velha")})

    assinatura = indexar(fontes, incluir_superadas=True, **amb)

    assert sorted(assinatura.fontes_indexadas) == ["pcdt-2024", "pcdt-2026"]
    assert assinatura.fontes_de_fora == {}


def test_assinatura_diz_se_o_indice_tem_superadas(tmp_path):
    """A sonda contrafactual das insulinas constrói o índice com a superada
    dentro. Sem o campo, os dois índices têm assinatura do mesmo formato e
    nada diz qual dos dois está carregado."""
    fontes = [fonte("pcdt-2026", substitui="pcdt-2024"), fonte("pcdt-2024")]
    amb = ambiente(tmp_path, {"pcdt-2026": documento("nova"), "pcdt-2024": documento("velha")})

    sem = indexar(fontes, **amb)
    com = indexar(fontes, incluir_superadas=True, **amb)

    assert sem.incluir_superadas is False
    assert com.incluir_superadas is True
    gravada = json.loads((amb["registro"] / "assinatura.json").read_text(encoding="utf-8"))
    assert gravada["incluir_superadas"] is True


def test_pedaco_recusado_fica_fora_da_matriz_e_dentro_da_assinatura(tmp_path):
    """A decisão do DECISOES.md: o recusado não entra, e fica registrado.

    O que se grava é o hash do texto, e não só o id, porque o id é posicional
    e muda de número quando o fatiador muda.
    """
    fontes = [fonte("pcdt-a")]
    amb = ambiente(tmp_path, {"pcdt-a": documento("alfa")}, chamar=ollama_falso(recusar=("Insulina",)))

    assinatura = indexar(fontes, **amb)
    estante = carregar(destino=amb["destino"], registro=amb["registro"],
                       modelo_em_execucao=amb["modelo_em_execucao"])

    assert assinatura.n_pedacos == 2
    assert estante.matriz.shape == (2, DIMENSAO)
    assert [r["id"] for r in assinatura.recusados] == ["pcdt-a#3.1"]
    recusado = assinatura.recusados[0]
    assert recusado["sha256_texto"] == digerir("Insulina NPH alfa " + "c" * 200)
    assert all("Insulina" not in p.texto for p in estante.pedacos)


def test_reindexar_corpus_intacto_da_arquivo_identico_sem_chamar_o_ollama(tmp_path):
    """A segunda indexação sai do cache: zero pedidos ao servidor, mesmos bytes."""
    fontes = [fonte("pcdt-a")]
    chamar = ollama_falso()
    amb = ambiente(tmp_path, {"pcdt-a": documento("alfa")}, chamar=chamar)

    indexar(fontes, **amb)
    pedidos_da_primeira = len(chamar.chamadas)
    arquivos = sorted(list(amb["destino"].iterdir()) + list(amb["registro"].iterdir()))
    antes = {a.name: a.read_bytes() for a in arquivos}

    indexar(fontes, **amb)

    depois = {a.name: a.read_bytes() for a in arquivos}
    assert pedidos_da_primeira > 0
    # O único pedido da segunda passada é o descarregamento do modelo.
    assert len(chamar.chamadas) == pedidos_da_primeira + 1
    assert chamar.chamadas[-1]["input"] == [""]
    assert antes == depois


def test_cache_nao_atravessa_modelo(tmp_path):
    """A chave do cache leva o nome do modelo: vetor do bge-m3 não serve ao E5."""
    fontes = [fonte("pcdt-a")]
    chamar = ollama_falso()
    amb = ambiente(tmp_path, {"pcdt-a": documento("alfa")}, chamar=chamar)

    indexar(fontes, **amb)
    pedidos = len(chamar.chamadas)
    indexar(fontes, modelo="outro-modelo", **amb)

    assert len(chamar.chamadas) > pedidos + 1


def test_cache_nao_serve_vetor_de_outro_digest(tmp_path):
    """Um `ollama pull` que troque o digest mantém o nome. Se a chave do cache
    só levasse o nome, a reindexação reaproveitaria todos os vetores velhos, a
    assinatura gravaria o digest novo em cima deles e `carregar` passaria."""
    fontes = [fonte("pcdt-a")]
    chamar = ollama_falso()
    amb = ambiente(tmp_path, {"pcdt-a": documento("alfa")}, chamar=chamar, digest="digest-a")

    indexar(fontes, **amb)
    pedidos = len(chamar.chamadas)
    amb["modelo_em_execucao"] = lambda: ("bge-m3", "digest-b", "0.34.0")
    indexar(fontes, **amb)

    assert len(chamar.chamadas) > pedidos + 1


def test_digest_vazio_e_recusado_ao_indexar(tmp_path):
    """Sem o modelo na lista do servidor o digest vem vazio, e vazio bate com
    vazio: o índice seria assinado por modelo nenhum e carregaria sempre."""
    fontes = [fonte("pcdt-a")]
    amb = ambiente(tmp_path, {"pcdt-a": documento("alfa")}, digest="")

    with pytest.raises(IndiceDeOutroModelo):
        indexar(fontes, **amb)


# --- o que se recusa a carregar ---------------------------------------------


def test_recusa_carregar_indice_desalinhado(tmp_path):
    """Matriz com uma linha a menos, e lista de ids que não bate com a assinatura."""
    fontes = [fonte("pcdt-a")]
    amb = ambiente(tmp_path, {"pcdt-a": documento("alfa")})
    indexar(fontes, **amb)
    carregar_ = lambda: carregar(destino=amb["destino"], registro=amb["registro"],
                                 modelo_em_execucao=amb["modelo_em_execucao"])

    matriz = np.load(amb["destino"] / "vetores.npy")
    np.save(amb["destino"] / "vetores.npy", matriz[:-1])
    with pytest.raises(IndiceDesalinhado):
        carregar_()

    np.save(amb["destino"] / "vetores.npy", matriz)
    carregar_()  # volta a carregar, para provar que o próximo erro é o outro

    registro = json.loads((amb["registro"] / "registro.json").read_text(encoding="utf-8"))
    registro["pedacos"][0]["id"] = "pcdt-a#1.9"
    (amb["registro"] / "registro.json").write_text(json.dumps(registro), encoding="utf-8")
    with pytest.raises(IndiceDesalinhado):
        carregar_()

    # Registro e pedaços adulterados de forma COERENTE, que é o que sobra quando
    # alguém regenera os dois e esquece a assinatura: só o hash da lista de ids
    # gravado nela pode acusar. Sem este caso, a conferência do sha256_ids
    # podia ser apagada do código com o teste verde.
    pesado = amb["destino"] / "pedacos.jsonl"
    linhas = pesado.read_text(encoding="utf-8").splitlines()
    primeira = json.loads(linhas[0])
    primeira["id"] = "pcdt-a#1.9"
    linhas[0] = json.dumps(primeira, ensure_ascii=False)
    pesado.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    with pytest.raises(IndiceDesalinhado):
        carregar_()


def _carregar(amb):
    return carregar(destino=amb["destino"], registro=amb["registro"],
                    modelo_em_execucao=amb["modelo_em_execucao"])


def test_texto_trocado_com_os_ids_intactos_e_recusado(tmp_path):
    """Só o `sha256_pedacos` acusa: o registro e a lista de ids continuam iguais.

    É o caso "regenerei o pesado com outro fatiador e esqueci o leve". A
    assinatura gravava esse hash desde o primeiro dia e `carregar` nunca o lia.
    """
    fontes = [fonte("pcdt-a")]
    amb = ambiente(tmp_path, {"pcdt-a": documento("alfa")})
    indexar(fontes, **amb)

    pesado = amb["destino"] / "pedacos.jsonl"
    linhas = pesado.read_text(encoding="utf-8").splitlines()
    primeira = json.loads(linhas[0])
    primeira["texto"] = "outro texto, mesmo id"
    linhas[0] = json.dumps(primeira, ensure_ascii=False)
    pesado.write_text("\n".join(linhas) + "\n", encoding="utf-8")

    with pytest.raises(IndiceDesalinhado):
        _carregar(amb)


def test_matriz_trocada_com_o_mesmo_tamanho_e_recusada(tmp_path):
    """Só o `sha256_vetores` acusa: o número de linhas bate e os ids também."""
    fontes = [fonte("pcdt-a")]
    amb = ambiente(tmp_path, {"pcdt-a": documento("alfa")})
    indexar(fontes, **amb)

    matriz = np.load(amb["destino"] / "vetores.npy")
    matriz[0, 0] += 0.01
    np.save(amb["destino"] / "vetores.npy", matriz)

    with pytest.raises(IndiceDesalinhado):
        _carregar(amb)


def test_indice_de_outra_versao_do_tokenizador_e_recusado(tmp_path, monkeypatch):
    """O BM25 é reconstruído ao carregar com o tokenizador de agora; se ele não
    é o que assinou o índice, a estatística gravada na assinatura mente."""
    from geracao_ancorada.estante import lexico

    fontes = [fonte("pcdt-a")]
    amb = ambiente(tmp_path, {"pcdt-a": documento("alfa")})
    indexar(fontes, **amb)
    monkeypatch.setattr(lexico, "VERSAO", "999")

    with pytest.raises(IndiceDesalinhado):
        _carregar(amb)


def test_indice_de_outro_modelo_e_recusado(tmp_path):
    """O bge-m3 e o E5 têm as mesmas 1.024 colunas; só o digest acusa a troca."""
    fontes = [fonte("pcdt-a")]
    amb = ambiente(tmp_path, {"pcdt-a": documento("alfa")}, digest="digest-do-bge")
    indexar(fontes, **amb)

    with pytest.raises(IndiceDeOutroModelo):
        carregar(destino=amb["destino"], registro=amb["registro"],
                 modelo_em_execucao=lambda: ("bge-m3", "digest-do-e5", "0.34.0"))


# --- o que o repositório público pode conter --------------------------------


def test_registro_publico_nao_contem_texto_de_fonte(tmp_path):
    """Nenhum campo se chama texto e nenhum valor passa de 120 caracteres."""
    fontes = [fonte("pcdt-a")]
    amb = ambiente(tmp_path, {"pcdt-a": documento("alfa")})
    indexar(fontes, **amb)

    registro = json.loads((amb["registro"] / "registro.json").read_text(encoding="utf-8"))

    assert registro["pedacos"]
    for linha in registro["pedacos"]:
        assert "texto" not in linha
        assert all(len(str(v)) <= 120 for v in linha.values()), linha


def test_gitignore_cobre_o_destino_pesado():
    """O destino com texto de fonte mora onde o .gitignore já cobre."""
    ignorados = (RAIZ / ".gitignore").read_text(encoding="utf-8").split()

    relativo = indice.DESTINO.relative_to(RAIZ).as_posix()

    assert any(relativo.startswith(regra.rstrip("/")) for regra in ignorados), relativo
    assert indice.REGISTRO.relative_to(RAIZ).as_posix() == "indice"


def test_assinatura_carrega_o_que_a_monografia_precisa_declarar(tmp_path):
    fontes = [fonte("pcdt-a")]
    amb = ambiente(tmp_path, {"pcdt-a": documento("alfa")})

    indexar(fontes, **amb)
    dados = json.loads((amb["registro"] / "assinatura.json").read_text(encoding="utf-8"))

    for campo in ("modelo", "digest_modelo", "dimensao", "versao_ollama", "k1", "b",
                  "versao_tokenizador", "commit_fatiador", "data", "sha256_ids",
                  "sha256_vetores", "sha256_pedacos"):
        assert campo in dados, campo
    assert dados["dimensao"] == DIMENSAO
    assert dados["commit_fatiador"] == "abc1234"
    assert dados["fusao"] == {
        "constante_rrf": 60,
        "garantidos_por_braco": 1,
        "profundidade": 50,
        "n_entregue": 8,
    }


# --- o dado real --------------------------------------------------------------


@pytest.mark.corpus
@pytest.mark.gpu
def test_a_indexacao_dos_onze_pdf_da_a_assinatura_medida():
    """A primeira indexação completa, contra o Ollama e os PDF de verdade.

    Os números são os do DECISOES.md: 4.280 indexáveis, dos quais o servidor
    recusa 10, todos tabela, seis do termo de esclarecimento do protocolo de
    hipertensão e quatro do apêndice 2 do de asma. A fonte de fora é a
    superada. Roda sobre os destinos reais, para que o registro versionado
    seja o mesmo que este teste confere.
    """
    from geracao_ancorada.fontes.manifesto import carregar_manifesto

    if not indice.CACHE_PDF.exists():
        pytest.skip("corpus não baixado")

    assinatura = indexar(carregar_manifesto(RAIZ / "fontes" / "manifesto.yaml"))
    estante = carregar()

    assert assinatura.n_pedacos == 4288
    assert assinatura.fontes_de_fora == {"pcdt-dm2-2024": "superada"}
    assert len(assinatura.fontes_indexadas) == 10
    assert len(assinatura.recusados) == 10
    assert sorted(r["id"].split("#")[1].split("/")[0] for r in assinatura.recusados) == (
        ["ap-ndice-2"] * 4 + ["termo-de-esclarecimento-"] * 6
    )
    assert estante.matriz.shape == (4288, DIMENSAO)
    assert estante.bm25.n_documentos == 4288

    registro = json.loads((indice.REGISTRO / "registro.json").read_text(encoding="utf-8"))
    for linha in registro["pedacos"]:
        assert "texto" not in linha
        assert all(len(str(v)) <= 120 for v in linha.values()), linha["id"]


def test_o_commit_e_lido_antes_de_o_indice_tocar_no_disco(tmp_path):
    """O registro é versionado, e o índice o reescreve ao indexar. Lido depois
    da escrita, o `git describe --dirty` via a própria saída do índice como
    mudança sem commit, e a reindexação de 15/09 saiu assinada "-dirty" com a
    árvore limpa. O commit tem que ser lido antes de qualquer escrita."""
    fontes = [fonte("pcdt-2026", substitui="pcdt-2024"), fonte("pcdt-2024")]
    amb = ambiente(tmp_path, {"pcdt-2026": documento("nova"), "pcdt-2024": documento("velha")})
    indexar(fontes, **amb)
    registro = amb["registro"] / "registro.json"
    limpo = registro.read_bytes()

    def git_describe():
        return "abc1234" if registro.read_bytes() == limpo else "abc1234-dirty"

    assinatura = indexar(fontes, incluir_superadas=True, **{**amb, "commit": git_describe})

    assert assinatura.commit_fatiador == "abc1234"
