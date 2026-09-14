import pytest

from geracao_ancorada.fatiamento.extracao import Bloco
from geracao_ancorada.fatiamento.pedacos import TETO, fatiar
from geracao_ancorada.fontes.manifesto import Fonte


@pytest.fixture
def fonte():
    return Fonte(
        id="pcdt-exemplo",
        titulo="Protocolo Clínico de Exemplo",
        orgao="MS/CONITEC",
        ano=2026,
        url="https://exemplo.invalido/pcdt.pdf",
        base_legal="Portaria de exemplo",
        redistribuivel=True,
        areas=["clinica-medica"],
    )


def texto(pagina, corpo):
    return Bloco(pagina=pagina, tipo="texto", texto=corpo)


def test_cada_secao_vira_pedaco_com_a_pagina_de_origem(fonte):
    blocos = [
        texto(3, "1. INTRODUÇÃO"),
        texto(3, "O diabete melito tipo 2 " + "x" * 200),
        texto(4, "2. DIAGNÓSTICO"),
        texto(4, "O diagnóstico laboratorial " + "y" * 200),
    ]
    pedacos = fatiar(fonte, blocos)

    assert [p.secao for p in pedacos] == ["1", "2"]
    assert [p.titulo_secao for p in pedacos] == ["INTRODUÇÃO", "DIAGNÓSTICO"]
    assert pedacos[0].pagina_inicial == pedacos[0].pagina_final == 3
    assert pedacos[1].pagina_inicial == pedacos[1].pagina_final == 4


def test_secao_longa_e_quebrada_mas_cada_parte_guarda_a_propria_pagina(fonte):
    blocos = [texto(1, "1. INTRODUÇÃO")]
    blocos += [texto(pagina, "z" * 900) for pagina in (10, 11, 12, 13)]

    pedacos = fatiar(fonte, blocos)

    assert len(pedacos) > 1
    assert all(len(p.texto) <= TETO for p in pedacos)
    # Nenhum pedaço pode declarar uma faixa que não seja a dele.
    assert all(p.pagina_final - p.pagina_inicial <= 1 for p in pedacos)
    assert pedacos[0].pagina_inicial == 10
    assert pedacos[-1].pagina_final == 13


def test_pedacos_da_mesma_secao_tem_id_diferente(fonte):
    """O id é a chave do registro e do cache do índice, e tem que ser único.

    A seção quebrada pelo teto é onde dois pedaços dividem fonte, parte e
    seção, e só o contador os separa. Não havia teste que exigisse isso, e
    trocar o contador por um número fixo deixava 4.280 pedaços do corpus com
    1.653 id distintos sem nada ficar vermelho.
    """
    blocos = [texto(1, "1. INTRODUÇÃO")]
    blocos += [texto(pagina, "z" * 900) for pagina in (10, 11, 12)]

    pedacos = fatiar(fonte, blocos)

    assert len(pedacos) > 1
    assert len({p.id for p in pedacos}) == len(pedacos)


@pytest.mark.corpus
def test_nenhum_id_se_repete_no_corpus(pedacos_do_corpus):
    """A mesma exigência, sobre os onze PDF e não sobre a fixtura."""
    ids = [p.id for p in pedacos_do_corpus]

    assert len(set(ids)) == len(ids)


def test_tabela_nao_se_funde_com_o_texto_ao_lado(fonte):
    blocos = [
        texto(7, "1. RASTREAMENTO"),
        texto(7, "Recomenda-se rastrear " + "a" * 200),
        Bloco(pagina=7, tipo="tabela", texto="Idade | Conduta\n45 anos | Rastrear"),
    ]
    pedacos = fatiar(fonte, blocos)

    tipos = {p.tipo for p in pedacos}
    assert tipos == {"texto", "tabela"}
    tabela = next(p for p in pedacos if p.tipo == "tabela")
    assert "45 anos" in tabela.texto and "Recomenda-se" not in tabela.texto


def test_cabecalho_sustenta_o_pedaco_fora_do_documento(fonte):
    blocos = [texto(9, "1. INTRODUÇÃO"), texto(9, "b" * 200)]
    pedaco = fatiar(fonte, blocos)[0]

    cabecalho = pedaco.cabecalho
    assert "Protocolo Clínico de Exemplo" in cabecalho
    assert "2026" in cabecalho
    assert "1. INTRODUÇÃO" in cabecalho
    assert "p. 9" in cabecalho
    assert pedaco.ancorado.startswith(cabecalho)


def test_referencias_ficam_marcadas_e_nao_viram_secao(fonte):
    """A suspensão depois de REFERÊNCIAS, e a fixtura tem que chegar nela.

    A primeira versão numerava as referências 15 e 16 logo depois da seção 2,
    e a regra da sequência já rejeita 15 depois de 2 antes de a suspensão ser
    consultada: o teste ficava verde com a suspensão apagada do código. Sem
    ela, 438 pedaços de bibliografia dos PCDT entram no índice.

    A lista de verdade começa em "1.", e o que a suspensão segura é a
    referência cujo número CONTINUA a seção de REFERÊNCIAS. Aqui a seção é a 2
    e a referência 3 a continua. Ela ocupa duas linhas, como no PDF, porque a
    segunda linha é o conteúdo que a "seção 3" ganharia.
    """
    blocos = [
        texto(1, "1. INTRODUÇÃO"),
        texto(1, "c" * 200),
        texto(40, "2. REFERÊNCIAS"),
        texto(40, "1. Padhi S. Type II diabetes mellitus: a review. " + "d" * 160),
        texto(40, "2. Brasil. Ministério da Saúde. Portaria nº 62. " + "e" * 160),
        texto(41, "3. American Diabetes Association. Standards of Care in Diabetes."),
        texto(41, "Diabetes Care. 2024;47(Suppl 1):S1-S4. " + "f" * 160),
    ]
    pedacos = fatiar(fonte, blocos)

    uteis = [p for p in pedacos if not p.descartavel]
    assert [p.secao for p in uteis] == ["1"]
    assert all(p.secao == "2" for p in pedacos if p.descartavel)
    assert any("Standards of Care" in p.texto for p in pedacos if p.descartavel)


def test_marco_reinicia_a_numeracao(fonte):
    blocos = [
        texto(1, "1. INTRODUÇÃO"),
        texto(1, "f" * 200),
        texto(70, "APÊNDICE 1 – METODOLOGIA"),
        texto(70, "1. Escopo e finalidade"),
        texto(70, "g" * 200),
    ]
    pedacos = fatiar(fonte, blocos)

    apendice = [p for p in pedacos if p.parte.startswith("APÊNDICE")]
    assert apendice and apendice[0].secao == "1"
    assert apendice[0].titulo_secao == "Escopo e finalidade"
    assert "APÊNDICE 1" in apendice[0].cabecalho


def test_fragmento_curto_demais_nao_sobrevive_sozinho(fonte):
    """O piso varre o fragmento solto, e ele tem que estar mesmo SOLTO.

    A fixtura precisa deixar o fragmento num pedaço que não ABRE a seção, senão
    o piso nem chega a ser consultado: quem abre seção fica por curto que seja,
    porque é todo o conteúdo que aquela seção tem. Aqui a seção já foi aberta
    por um corpo que passa do teto, e o "2017;" — que é o rodapé de uma
    referência, a forma em que isso aparece nos PCDT — vem depois dele.
    """
    blocos = [
        texto(1, "1. INTRODUÇÃO"),
        texto(1, "O diabete melito tipo 2 " + "x" * TETO),
        texto(1, "2017;"),
    ]
    pedacos = fatiar(fonte, blocos)

    assert len(pedacos) == 1
    assert "2017;" not in pedacos[0].texto


def test_fragmento_curto_que_abre_a_secao_fica(fonte):
    """A exceção do piso, que é o que o impede de apagar seção curta."""
    blocos = [
        texto(1, "1. INTRODUÇÃO"),
        texto(1, "O diabete melito tipo 2 " + "x" * 300),
        texto(1, "2. SIGLAS"),
        texto(1, "DM2, HAS"),
    ]
    pedacos = fatiar(fonte, blocos)

    assert [p.secao for p in pedacos] == ["1", "2"]
    assert pedacos[1].texto == "DM2, HAS"


def test_secao_curta_nao_engole_o_conteudo_da_seguinte(fonte):
    # No pcdt-has-2025 a seção 8 tem 116 caracteres. Ao colar no vizinho, ela
    # levava junto a seção 9 inteira e mantinha o rótulo "8": a 9 sumia e o
    # texto dela saía citado como se fosse critério de inclusão.
    blocos = [
        texto(21, "1. CRITÉRIOS DE INCLUSÃO"),
        texto(21, "Devem ser incluídos indivíduos com diagnóstico confirmado."),
        texto(21, "2. CRITÉRIOS DE EXCLUSÃO"),
        texto(21, "Pacientes cujo diagnóstico foi descartado " + "x" * 200),
    ]
    pedacos = fatiar(fonte, blocos)

    assert [p.secao for p in pedacos] == ["1", "2"]
    oito = next(p for p in pedacos if p.secao == "1")
    assert "descartado" not in oito.texto


def test_secao_curta_sobrevive_mesmo_sem_vizinho_para_colar(fonte):
    # Consequência da regra acima: se não pode colar na seguinte, a seção curta
    # não pode simplesmente desaparecer — ela é o conteúdo inteiro da seção.
    blocos = [
        texto(21, "1. CRITÉRIOS DE INCLUSÃO"),
        texto(21, "Devem ser incluídos indivíduos com diagnóstico confirmado."),
        texto(22, "2. CRITÉRIOS DE EXCLUSÃO"),
        texto(22, "Pacientes cujo diagnóstico foi descartado " + "x" * 200),
    ]
    pedacos = fatiar(fonte, blocos)

    oito = [p for p in pedacos if p.secao == "1"]
    assert len(oito) == 1
    assert "Devem ser incluídos" in oito[0].texto


def test_referencias_suspendem_mesmo_com_a_numeracao_reiniciada(fonte):
    # No pcdt-dpoc-2025 a linha de corpo "Termo de Esclarecimento e
    # Responsabilidade (TER)." dispara o marco e reinicia a numeração logo antes
    # de "11. REFERÊNCIAS". A numerada é então rejeitada, a suspensão não liga, e
    # a lista de referências — que começa em "1." — assume a espinha inteira.
    blocos = [
        texto(1, "1. INTRODUÇÃO"),
        texto(1, "A DPOC é uma doença " + "a" * 200),
        texto(28, "Termo de Esclarecimento e Responsabilidade (TER)."),
        texto(28, "11. REFERÊNCIAS"),
        texto(28, "1. Celli B, Fabbri L, Criner G, et al. Definition of COPD."),
        texto(28, "Am J Respir Crit Care Med. 2022;206(11):1317-1325. " + "b" * 150),
        texto(29, "2. Agustí A, Melén E, DeMeo DL, et al. Pathogenesis of COPD."),
        texto(29, "Lancet Respir Med. 2022;10(5):512-524. " + "c" * 150),
    ]
    pedacos = fatiar(fonte, blocos)

    uteis = [p for p in pedacos if not p.descartavel]
    assert [p.secao for p in uteis] == ["1"]
    assert uteis[0].titulo_secao == "INTRODUÇÃO"
    citados = [p for p in pedacos if "Respir" in p.texto]
    assert citados and all(p.descartavel for p in citados)


def test_referencias_sem_numero_nao_suspendem_o_resto_do_livro(fonte):
    # O Guia de Vigilância é um livro de capítulos: cada um termina em
    # "REFERÊNCIAS" (sem número, em ABNT) e o seguinte começa sem numeração
    # nenhuma. Suspender ali descartava o livro inteiro a partir do primeiro
    # capítulo — tabela de dose de tuberculose incluída. E não é preciso: só a
    # lista NUMERADA pode ser confundida com a espinha de seções.
    blocos = [
        texto(27, "REFERÊNCIAS"),
        texto(27, "BRASIL. Ministério da Saúde. Critérios de definição de casos de aids. " + "a" * 140),
        texto(30, "HEPATITES VIRAIS"),
        texto(30, "CID-10: B15 - B19.9"),
        texto(30, "As hepatites virais são doenças causadas por vírus hepatotrópicos " + "b" * 140),
    ]
    pedacos = fatiar(fonte, blocos)

    hepatites = [p for p in pedacos if "hepatites virais são doenças" in p.texto]
    assert hepatites, "o capítulo seguinte sumiu"
    assert not hepatites[0].descartavel


def test_indice_no_comeco_nao_consome_a_espinha_do_corpo(fonte):
    # A Instrução Normativa do Calendário abre com um índice: 21 títulos
    # numerados seguidos, todos na página 1, sem conteúdo entre eles. O índice
    # consumia a numeração inteira; o corpo, que recomeça em "1.", era rejeitado
    # e cada pedaço herdava o rótulo da ÚLTIMA linha do índice — foi assim que o
    # texto da varicela saiu etiquetado como a seção da dT.
    indice = [
        texto(1, "1. Vacina hepatite B (recombinante) - HB"),
        texto(1, "2. Vacina BCG (atenuada) - BCG"),
        texto(1, "3. Vacina influenza trivalente (inativada) - INF3"),
        texto(1, "4. Vacina varicela monovalente (atenuada) - VZ"),
        texto(1, "5. Vacina adsorvida difteria e tétano adulto - dT"),
    ]
    corpo = [
        texto(2, "Esta Instrução Normativa estabelece o calendário " + "a" * 200),
        texto(10, "1. Vacina hepatite B (recombinante) - HB"),
        texto(10, "A vacina hepatite B deve ser administrada ao nascer " + "b" * 200),
        texto(12, "2. Vacina BCG (atenuada) - BCG"),
        texto(12, "A vacina BCG deve ser administrada em dose única " + "c" * 200),
        texto(14, "3. Vacina influenza trivalente (inativada) - INF3"),
        texto(14, "A vacina influenza é oferecida na campanha anual " + "d" * 200),
        texto(20, "4. Vacina varicela monovalente (atenuada) - VZ"),
        texto(20, "Imunoglobulina humana antivaricela-zóster (IGHVZ) " + "e" * 200),
    ]
    pedacos = fatiar(fonte, indice + corpo)

    zoster = next(p for p in pedacos if "IGHVZ" in p.texto)
    assert zoster.secao == "4", f"herdou a seção {zoster.secao}: {zoster.titulo_secao}"
    assert "varicela" in zoster.titulo_secao.lower()


def test_pagina_que_lista_anexos_nao_abre_parte(fonte):
    # No manual de tuberculose a página 18 é a LISTA de anexos: sete linhas
    # "anexo i – ...", "anexo ii – ...", uma atrás da outra. Cada uma disparava
    # a regra de marco, a última grudava, e o corpo inteiro do manual saía
    # etiquetado como "anexo XIV – Ficha de acompanhamento da tomada diária da
    # medicação" — 252 pedaços dizendo que vieram de um anexo que não é o deles.
    # Um anexo de verdade ocupa a página; três na mesma página são a lista dela.
    blocos = [
        texto(18, "anexo i – segurança dos fármacos antitb em gestantes 177"),
        texto(18, "anexo ii – segurança dos fármacos antitb em lactantes 178"),
        texto(18, "anexo iii – ajuste dos medicamentos em nefropatas 179"),
        texto(18, "anexo iv – tabela posológica dos medicamentos para adultos"),
        texto(24, "A tuberculose é uma doença infecciosa e transmissível " + "a" * 200),
        texto(178, "ANEXO I – SEGURANÇA DOS FÁRMACOS ANTITB EM GESTANTES"),
        texto(178, "A isoniazida é segura na gestação e deve ser mantida " + "b" * 200),
    ]
    pedacos = fatiar(fonte, blocos)

    corpo = next(p for p in pedacos if "doença infecciosa" in p.texto)
    assert corpo.parte == "", f"corpo herdou a parte {corpo.parte!r}"

    anexo = next(p for p in pedacos if "isoniazida" in p.texto)
    assert anexo.parte.startswith("ANEXO I"), f"o anexo de verdade virou {anexo.parte!r}"


def test_capitulo_do_guia_encerra_o_anexo_anterior(fonte):
    # No Guia de Vigilância v2 o rótulo 'ANEXO C' cobria da página 328 à 560,
    # 278 pedaços, atravessando capítulos inteiros — porque nada encerrava o
    # anexo. O capítulo seguinte é que encerra, e ele se identifica pelo par
    # título em caixa alta + "CID-10:".
    blocos = [
        texto(268, "ANEXO – FICHA DE SOLICITAÇÃO DE ANTIFÚNGICOS"),
        texto(268, "Preencher a ficha em duas vias e encaminhar " + "a" * 200),
        texto(280, "HEPATITES VIRAIS"),
        texto(280, "CID-10: B15 – B19.9"),
        texto(280, "As hepatites virais são doenças causadas por vírus " + "b" * 200),
    ]
    pedacos = fatiar(fonte, blocos)

    hepatites = next(p for p in pedacos if "hepatites virais são" in p.texto)
    assert hepatites.parte == "HEPATITES VIRAIS", f"herdou {hepatites.parte!r}"


def test_lista_numerada_nao_vira_secao_em_livro_de_capitulo(fonte):
    # O Guia de Vigilância não tem espinha numerada. A regra da sequência —
    # que é o que separa título de referência nos PCDT — não distingue título
    # de LISTA, porque uma lista também anda 1, 2, 3. Sem espinha para
    # disputar, a lista vencia sozinha: 461 pedaços dos três volumes saíam
    # rotulados com item de lista, ficha catalográfica ou trecho de
    # bibliografia, e esse rótulo ia impresso na citação.
    blocos = [
        texto(51, "SARAMPO"),
        texto(51, "CID-10: B05"),
        texto(51, "} TESTES DA TRIAGEM NEONATAL"),
        texto(51, "O diagnóstico também ocorre por meio dos testes " + "a" * 200),
        texto(51, "1. Teste do Pezinho: realizado entre o 3° e o 5° dia de vida"),
        texto(51, "2. Teste do Olhinho: também chamado de teste do reflexo vermelho"),
    ]
    pedacos = fatiar(fonte, blocos)

    assert all(p.secao == "" for p in pedacos), [p.secao for p in pedacos]
    assert all("Teste do Pezinho" not in p.titulo_secao for p in pedacos)


def test_subtitulo_marcado_vira_a_secao_do_pedaco(fonte):
    # É este o rótulo que a prova de residência pergunta: "qual o período de
    # incubação do sarampo". Sem ele o pedaço fica só com o capítulo.
    blocos = [
        texto(262, "SARAMPO"),
        texto(262, "CID-10: B05"),
        texto(262, "} DESCRIÇÃO"),
        texto(262, "Doença exantemática aguda de natureza viral " + "a" * 200),
        texto(265, "} PERÍODO DE INCUBAÇÃO"),
        texto(265, "Geralmente de dez dias, podendo variar de sete " + "b" * 200),
    ]
    pedacos = fatiar(fonte, blocos)

    incubacao = next(p for p in pedacos if "Geralmente de dez dias" in p.texto)
    assert incubacao.parte == "SARAMPO"
    assert incubacao.titulo_secao == "PERÍODO DE INCUBAÇÃO"
    assert "SARAMPO — PERÍODO DE INCUBAÇÃO" in incubacao.cabecalho


def test_secao_sem_numero_nao_imprime_ponto_solto_na_citacao(fonte):
    # A citação é o produto: "8.1. Metas terapêuticas" tem número, "PERÍODO DE
    # INCUBAÇÃO" não tem, e nenhuma das duas pode sair com pontuação sobrando.
    blocos = [
        texto(262, "SARAMPO"),
        texto(262, "CID-10: B05"),
        texto(262, "} MODO DE TRANSMISSÃO"),
        texto(262, "Diretamente de pessoa a pessoa, por secreções " + "a" * 200),
    ]
    pedaco = fatiar(fonte, blocos)[-1]
    assert ". MODO" not in pedaco.cabecalho
    assert "(sem seção)" not in pedaco.cabecalho


def test_a_espinha_numerada_do_pcdt_continua_valendo(fonte):
    # A regra nova vale para o livro de capítulo, e só. O PCDT não tem
    # capítulo, e a numeração dele segue sendo a espinha.
    blocos = [
        texto(3, "1. INTRODUÇÃO"),
        texto(3, "O diabete melito tipo 2 " + "x" * 200),
        texto(4, "2. DIAGNÓSTICO"),
        texto(4, "O diagnóstico laboratorial " + "y" * 200),
    ]
    pedacos = fatiar(fonte, blocos)
    assert [p.secao for p in pedacos] == ["1", "2"]


def test_bibliografia_do_guia_nao_gera_estrutura(fonte):
    # "Anexo V – Sistema Nacional de Vigilância Epidemiológica (SNVE)" é uma
    # LINHA DE CITAÇÃO ABNT dentro da lista de referências, e virava rótulo de
    # parte. No Guia o cabeçalho da bibliografia é o "REFERÊNCIAS" pelado, sem
    # número, e por isso escapava da regra que já existia para o PCDT.
    blocos = [
        texto(400, "CÓLERA"),
        texto(400, "CID-10: A00"),
        texto(400, "} DESCRIÇÃO"),
        texto(400, "Infecção intestinal aguda causada pela enterotoxina " + "a" * 200),
        texto(404, "REFERÊNCIAS"),
        texto(404, "BRASIL. Ministério da Saúde. Portaria de Consolidação n.º 4."),
        texto(404, "Anexo V – Sistema Nacional de Vigilância Epidemiológica (SNVE)"),
        texto(405, "DOENÇAS DIARREICAS AGUDAS"),
        texto(405, "CID-10: A08"),
        texto(405, "As doenças diarreicas agudas são um grupo de doenças " + "b" * 200),
    ]
    pedacos = fatiar(fonte, blocos)

    diarreicas = next(p for p in pedacos if "grupo de doenças" in p.texto)
    assert diarreicas.parte == "DOENÇAS DIARREICAS AGUDAS", f"herdou {diarreicas.parte!r}"
    assert all(not p.parte.startswith("Anexo V") for p in pedacos)


def test_bibliografia_termina_no_proximo_subtitulo(fonte):
    # A suspensão da bibliografia fechava só no capítulo seguinte. Nos
    # primeiros 107 páginas do volume 1 não há capítulo com CID — são as
    # seções gerais (vigilância do óbito, anomalias congênitas, saúde do
    # trabalhador) —, então a suspensão atravessava a seção inteira de
    # depois: de p34 a p46, de p46 a p61, e a partir de p444 até o fim do
    # volume. O subtítulo marcado também fecha, e bibliografia não tem
    # subtítulo marcado.
    blocos = [
        texto(34, "REFERÊNCIAS"),
        texto(34, "BRASIL. Decreto n.º 78.231, de 12 de agosto de 1976. " + "a" * 200),
        texto(40, "} OBJETIVO GERAL"),
        texto(40, "Reduzir a mortalidade materna por meio da vigilância " + "b" * 200),
        # O capítulo com CID só aparece na página 108 do volume: é ele que
        # identifica a família do documento, e as seções gerais vêm antes.
        texto(108, "DOENÇA MENINGOCÓCICA"),
        texto(108, "CID-10: A39"),
        texto(108, "Infecção bacteriana aguda causada pela Neisseria " + "c" * 200),
    ]
    pedacos = fatiar(fonte, blocos)

    objetivo = next(p for p in pedacos if "Reduzir a mortalidade" in p.texto)
    assert not objetivo.descartavel, "conteúdo clínico saiu como bibliografia"
    assert objetivo.titulo_secao == "OBJETIVO GERAL"

    citacao = next(p for p in pedacos if "Decreto" in p.texto)
    assert citacao.descartavel, "a bibliografia voltou a concorrer na busca"


def test_bibliografia_termina_tambem_no_titulo_sem_marca(fonte):
    # As seções gerais do guia (farmacovigilância, saúde do trabalhador,
    # saúde ambiental) não usam o marcador: o título vem pelado, em caixa
    # alta. Sem fechar por ele, a bibliografia de uma seção geral engolia a
    # seguinte inteira — 11 páginas de farmacovigilância saíram como
    # bibliografia. Título não termina em ponto; "SMS, 2007." é sobra de
    # citação e continua dentro.
    blocos = [
        texto(96, "REFERÊNCIAS"),
        texto(96, "OPAS. Boas práticas de farmacovigilância. Brasília, DF. " + "a" * 200),
        texto(96, "SMS, 2007."),
        texto(98, "FARMACOVIGILÂNCIA"),
        texto(99, "A principal estratégia é a farmacovigilância passiva " + "b" * 200),
        texto(108, "DOENÇA MENINGOCÓCICA"),
        texto(108, "CID-10: A39"),
        texto(108, "Infecção bacteriana aguda causada pela Neisseria " + "c" * 200),
    ]
    pedacos = fatiar(fonte, blocos)

    corpo = next(p for p in pedacos if "farmacovigilância passiva" in p.texto)
    assert not corpo.descartavel, "conteúdo clínico saiu como bibliografia"

    citacao = next(p for p in pedacos if "Boas práticas" in p.texto)
    assert citacao.descartavel and "SMS, 2007." in citacao.texto
