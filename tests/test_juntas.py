from geracao_ancorada.fatiamento.juntas import (
    abre_referencias,
    capitulo,
    detectar,
    marco,
    subtitulo,
    sucede,
)


def test_sequencia_comeca_em_um():
    assert sucede(None, (1,))
    assert not sucede(None, (2,))
    assert not sucede(None, (1, 1))


def test_desce_um_nivel():
    assert sucede((5,), (5, 1))
    assert sucede((8, 3, 4), (8, 3, 4, 1))
    assert not sucede((5,), (5, 2))


def test_avanca_no_mesmo_nivel():
    assert sucede((5, 1), (5, 2))
    assert sucede((8, 10), (8, 11))
    assert not sucede((5, 1), (5, 3))


def test_sobe_para_o_proximo_irmao_de_qualquer_ancestral():
    assert sucede((8, 3, 4, 3), (8, 3, 5))
    assert sucede((8, 3, 4, 3), (8, 4))
    assert sucede((8, 3, 4, 3), (9,))
    assert not sucede((8, 3, 4, 3), (11,))


def test_titulo_de_secao_e_aceito():
    junta = detectar("8.1. Metas terapêuticas", (8,))
    assert junta is not None
    assert junta.numero == "8.1"
    assert junta.titulo == "Metas terapêuticas"


def test_quebra_de_paragrafo_com_numero_nao_e_titulo():
    # "4 U109,111,112. Quando a glicemia..." aparece no meio da seção 8.3.4.1.
    assert detectar("4 U109,111,112. Quando a glicemia de jejum", (8, 3, 4, 1)) is None


def test_referencia_numerada_nao_continua_a_sequencia():
    # Logo depois de "14. REFERÊNCIAS" vem "1. ElSayed NA...".
    assert detectar("1. ElSayed NA, McCoy RG, Aleppo G. Diabetes Care", (14,)) is None


def test_referencia_que_calharia_de_continuar_exige_o_ponto():
    # Nota de rodapé sem ponto depois do número não é título.
    assert detectar("1 HC-UFMG. Centro de telessaúde", None) is None


def test_referencias_sao_marcadas_descartaveis():
    junta = detectar("14. REFERÊNCIAS", (13,))
    assert junta is not None and junta.descartavel


def test_marco_de_parte():
    assert marco("APÊNDICE 1 – METODOLOGIA DE BUSCA") is not None
    assert marco("ANEXO") == "ANEXO"
    assert marco("Anexos foram consultados ao longo do processo de elaboração") is None


def test_variante_antiga_numera_sem_ponto():
    # O pcdt-dislipidemia-2019 usa o modelo velho: "1 INTRODUÇÃO", sem ponto.
    junta = detectar("1 INTRODUÇÃO", None)
    assert junta is not None
    assert junta.numero == "1" and junta.titulo == "INTRODUÇÃO"


def test_sem_ponto_so_vale_para_titulo_em_caixa_alta():
    # É a caixa alta que separa o título da nota de rodapé e do parágrafo
    # quebrado; sem ela, "1 HC-UFMG. Centro de telessaúde" viraria seção.
    assert detectar("1 Neste capítulo apresentamos o quadro clínico", None) is None
    assert detectar("2 pacientes foram excluídos da análise", (1,)) is None


def test_titulo_de_uma_letra_nao_e_titulo():
    # O piso de comprimento da variante sem ponto, que barra a sigla solta de
    # uma legenda ou de uma coluna numerada ("1 DM2") sem barrar a seção curta
    # de verdade ("1 SIGLAS").
    #
    # Ao contrário das outras regras deste arquivo, este piso NÃO sai de uma
    # grafia conferida: varri os onze PDF e nenhuma linha deles é decidida por
    # ele — afrouxá-lo para uma letra devolve o corpus inteiro igual, 4.499
    # pedaços idênticos. Ele fica como guarda declarada, e o teste existe para
    # que o número pare de poder mudar sozinho.
    assert detectar("1 DM2", None) is None
    junta = detectar("1 SIGLAS", None)
    assert junta is not None and junta.titulo == "SIGLAS"


def test_titulo_longo_com_ponto_continua_valendo():
    # No Calendário de Vacinação o título da seção passa de 90 caracteres:
    # "1. Vacina adsorvida difteria, tétano e pertussis acelular (dTpa)...".
    linha = (
        "1. Vacina adsorvida difteria, tétano e pertussis acelular (dTpa) e "
        "vacina hepatite B (recombinante) - HB"
    )
    assert len(linha) > 90
    junta = detectar(linha, None)
    assert junta is not None and junta.numero == "1"


def test_paragrafo_longo_em_caixa_alta_nao_e_titulo():
    # O limite de tamanho só faz sentido onde falta o ponto: é lá que um
    # parágrafo gritado poderia passar por título.
    linha = "2 " + "ESTE TEXTO EM CAIXA ALTA SEGUE POR MUITO TEMPO E NAO E UM TITULO " * 2
    assert detectar(linha, (1,)) is None


def test_divisor_de_parte_sozinho_na_linha_e_marco():
    # O manual de tuberculose separa as partes com uma página que traz só
    # "PARTE IV". Sem reconhecer isso, o rótulo do último anexo atravessa a
    # parte inteira: 144 pedaços saíram etiquetados como ANEXO VII.
    assert marco("PARTE IV") == "PARTE IV"
    assert marco("PARTE I") == "PARTE I"


def test_linha_de_sumario_e_de_prosa_nao_sao_divisor_de_parte():
    # No sumário a mesma palavra vem com o título e o número da página colados,
    # e no texto corrido ela aparece no meio da frase. Nem uma nem outra abre parte.
    assert marco("PARTE IV • ESTRATÉGIAS PROGRAMÁTICAS") is None
    assert marco("PARTE I • ASPECTOS BÁSICOS E EPIDEMIOLÓGICOS 25") is None
    assert marco("Parte II (diagnóstico), em que estão descritos os métodos") is None


def test_titulo_de_capitulo_do_guia_e_reconhecido_pelo_cid():
    # O Guia de Vigilância é livro de capítulos sem numeração: o que abre um
    # capítulo é o título em caixa alta seguido da linha "CID-10:". São 64
    # capítulos nos três volumes, todos com esse par.
    assert capitulo("HEPATITES VIRAIS", "CID-10: B15 – B19.9") == "HEPATITES VIRAIS"
    assert capitulo("HANSENÍASE", "CID-10: A30") == "HANSENÍASE"
    assert capitulo("PESTE", "Cid-10: A20") == "PESTE"


def test_mencao_a_cid_no_texto_corrido_nao_abre_capitulo():
    # "As mortes maternas são causadas por afecções do Capítulo X / CID-10,
    # exceção para..." — sem os dois-pontos e com a linha anterior em prosa.
    assert capitulo(
        "As mortes maternas são causadas por afecções do Capítulo X",
        "CID-10, exceção para mortes fora do período",
    ) is None
    assert capitulo("DOENÇA MENINGOCÓCICA", "A doença meningocócica é uma infecção") is None
    # As duas linhas acima são rejeitadas ANTES de chegar na regra dos
    # dois-pontos — a primeira porque o título tem minúscula, a segunda porque a
    # linha seguinte não traz "CID-10" nenhum. Quem isola a vírgula é este
    # terceiro caso: título em caixa alta, "CID-10" presente, e só a pontuação
    # separando a menção em prosa do cabeçalho de capítulo.
    assert capitulo("DOENÇA MENINGOCÓCICA", "CID-10, exceção para mortes fora") is None
    assert capitulo("DOENÇA MENINGOCÓCICA", "CID-10: A39") == "DOENÇA MENINGOCÓCICA"


def test_capitulo_vale_com_o_cid_no_fim_da_linha():
    # No volume 1 a influenza abre com "INFLUENZA SAZONAL" seguido de
    # "Influenza devida a vírus não identificado CID-10: J11": o código vem no
    # FIM da linha, não no começo. Era o único capítulo perdido nos três
    # volumes, e o rótulo do anexo anterior atravessava 41 páginas atrás dele.
    assert capitulo(
        "INFLUENZA SAZONAL",
        "Influenza devida a vírus não identificado CID-10: J11",
    ) == "INFLUENZA SAZONAL"


def test_titulo_de_capitulo_quebrado_em_duas_linhas_e_remontado():
    # "TOXOPLASMOSE ADQUIRIDA NA GESTAÇÃO" / "E TOXOPLASMOSE CONGÊNITA": o
    # título ocupa duas linhas e a citação saía com a metade de baixo.
    assert capitulo(
        "E TOXOPLASMOSE CONGÊNITA",
        "CID-10: O98.6 – doenças causadas por protozoários",
        "TOXOPLASMOSE ADQUIRIDA NA GESTAÇÃO",
    ) == "TOXOPLASMOSE ADQUIRIDA NA GESTAÇÃO E TOXOPLASMOSE CONGÊNITA"


def test_cabecalho_de_grupo_nao_e_remontado_no_titulo_de_capitulo():
    # "OUTRAS MENINGITES" / "MENINGITES BACTERIANAS" são duas coisas: a
    # primeira agrupa, a segunda é o capítulo. É a conjunção que marca a
    # continuação, não a simples vizinhança de duas linhas em caixa alta.
    assert capitulo(
        "MENINGITES BACTERIANAS",
        "CID-10: G00.0 – meningite por Haemophilus influenzae",
        "OUTRAS MENINGITES",
    ) == "MENINGITES BACTERIANAS"


def test_subtitulo_do_guia_e_a_linha_marcada_em_caixa_alta():
    # O Guia não numera seção: ele marca a subseção com uma chave e caixa
    # alta. São 1.325 delas nos três volumes, e são o eixo em que a prova de
    # residência pergunta.
    assert subtitulo("} PERÍODO DE INCUBAÇÃO") == "PERÍODO DE INCUBAÇÃO"
    assert subtitulo("} MODO DE TRANSMISSÃO") == "MODO DE TRANSMISSÃO"
    assert subtitulo("}AGENTE ETIOLÓGICO") == "AGENTE ETIOLÓGICO"


def test_item_de_lista_usa_a_mesma_chave_e_nao_e_subtitulo():
    # A mesma chave marca item de lista comum; o que separa é a caixa alta.
    assert subtitulo("} Coletar e registrar os dados de vacinação.") is None
    assert subtitulo("} Insuficiência renal oligúrica.") is None
    assert subtitulo("PERÍODO DE INCUBAÇÃO") is None


def test_referencias_sem_numero_sao_reconhecidas():
    # No Guia o cabeçalho da bibliografia é o "REFERÊNCIAS" pelado, 72 vezes.
    # Sem reconhecê-lo, linhas da própria bibliografia viram estrutura: foi
    # assim que "Anexo V – Sistema Nacional de Vigilância Epidemiológica",
    # que é uma linha de citação ABNT, virou rótulo de parte.
    assert abre_referencias("REFERÊNCIAS")
    assert abre_referencias("BIBLIOGRAFIA")
    assert not abre_referencias("REFERÊNCIAS BIBLIOGRÁFICAS CONSULTADAS AO LONGO DO TEXTO")
    assert not abre_referencias("Referências excluídas do fluxograma")
