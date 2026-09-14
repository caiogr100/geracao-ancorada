# Decisões do harness

Registro das escolhas que mudam o resultado do experimento e que não se
deduzem do código lendo. Cada entrada diz o que foi decidido, por que, qual
medição sustenta a decisão e o que foi descartado junto.

A regra de quando escrever aqui: se alguém pudesse olhar o código daqui a três
meses e perguntar "por que está assim?", a resposta mora nesta lista. Ajuste de
implementação, refatoração e conserto de defeito não entram.

## 2026-09-14 — Pedaço que o servidor de embedding recusa fica de fora, registrado

**Decisão.** O pedaço cujo texto passa do teto de contexto do servidor não entra
no índice, e a lista dos ausentes é gravada junto com o índice. O corpus não é
alterado para acomodá-los e o teto do servidor não é levantado.

**Por quê.** São dez pedaços de 4.280, todos do tipo tabela, e nenhum deles
carrega conteúdo que a prova pergunta: seis são o termo de esclarecimento e
responsabilidade do protocolo de hipertensão, que é o papel que o paciente
assina, e quatro são a pergunta de pesquisa do apêndice 2 do protocolo de asma.
As duas alternativas custam mais do que valem. Levantar o teto do servidor
transforma o índice em algo que depende de uma configuração de máquina, e o
índice precisa sair igual na máquina de desenvolvimento e na do experimento.
Partir os pedaços no fatiador muda o corpus, que congela em outubro com um hash,
e o hash é o que garante que o experimento é o mesmo em qualquer lugar.

**Medição que sustenta.** Busca binária no teto do servidor, nesta máquina, com
o modelo bge-m3 pelo Ollama 0.34.0: o servidor obedece o pedido de contexto para
baixo, com 512 devolvendo 512 tokens, e satura em torno de 2.048 para cima, com
4.096, 8.192 e 16.384 dando o mesmo resultado, embora o modelo declare 8.192. O
teto em caracteres depende da densidade do texto, cerca de 8.900 em prosa
corrida contra 3.300 numa tabela cheia de número e de barra vertical. Rodando os
4.280 pedaços indexáveis contra o servidor, dez voltaram recusados.

O que se manda ao servidor muda esse número, e a primeira versão desta entrada
errou por isso. Submetendo o `texto` do pedaço sozinho, nove voltam recusados;
submetendo o texto indexado, que é o que a indexação manda de verdade e leva a
moldura de procedência e a legenda da tabela, voltam dez. O décimo é o
`pcdt-asma-2021#ap-ndice-2/preambulo.41`, com 3.596 caracteres de texto e 3.667
já com a moldura: os 71 caracteres de moldura são o que o põe do outro lado do
teto. A moldura acrescenta 115 caracteres na mediana e até 259, então a fronteira
mora perto, e o número medido tem que sair do texto que a indexação manda.

**O que isso obriga.** A lista de recusados é produzida pelo código, e não pela
leitura da saída do terminal: `vetorizar_corpus` devolve as posições recusadas,
e a assinatura do índice as grava. Quando entrar documento novo no corpus, a
reindexação mostra a lista nova, e a comparação com a gravada é o que avisa se
o número cresceu.

**Ressalva, resolvida em 14/09.** A primeira medição do teto foi feita com a
placa de vídeo ocupada por outro programa, e ficava a dúvida de o teto ser falta
de memória em vez de configuração do servidor. A busca binária foi repetida com
a placa livre e deu o mesmo resultado: 512 devolve 512, 1.024 devolve 1.024,
2.048 devolve 2.048, e 4.096, 8.192 e 16.384 devolvem todos 2.048. O teto é do
servidor.

## 2026-09-14 — A identidade do pedaço no índice é o hash do texto, e o id é a etiqueta

**Decisão.** O registro e a assinatura gravam, para cada pedaço, o hash do
texto ao lado do id. O cache de vetores é endereçado por `sha256(modelo +
texto que entra no encoder)`. A comparação que avisa se a lista de recusados
mudou entre duas indexações se faz pelo hash, e não pelo id.

**Por quê.** O id é posicional: `pcdt-asma-2021#ap-ndice-2/preambulo.41` quer
dizer "o 41º pedaço daquele grupo". Basta o fatiador passar a cortar um pedaço
a mais antes dele para o mesmo texto virar `.42`, e uma lista de recusados
guardada por id acusaria uma diferença que não existe, ou deixaria passar uma
que existe. O hash do texto só muda quando o texto muda. O id continua sendo o
que a citação imprime e o que uma pessoa lê.

**Medição que sustenta.** Um teste que troca o contador do id por um número
fixo deixa 4.280 pedaços com 1.653 id distintos; nada no texto dos pedaços
muda. O hash de cada um continua distinto.

**O que isso obriga.** A matriz só tem os pedaços aceitos, alinhados linha a
linha com o registro; a assinatura guarda o hash da lista de ids e o hash do
arquivo de pedaços, e `carregar` se recusa quando os três divergem. Módulo
nenhum pode usar a posição na matriz como identidade fora de uma sessão.

**Ressalva conhecida.** A constante do RRF e a cota por braço, que o desenho
manda gravar na assinatura, ficam de fora até o módulo de fusão existir:
gravar parâmetro de código que ainda não existe é documentar ficção.

## 2026-09-14 — O servidor recusa em voz alta em vez de cortar em silêncio

**Decisão.** Todo pedido de embedding sai com `truncate: false`.

**Por quê.** Com `truncate: true`, que é o padrão, o servidor corta o que passa
do teto e devolve 200 com um vetor de aparência normal. A tabela cortada no meio
entra no índice, pontua na busca e não contém a linha da dose, e nada no
caminho acusa isso. Nenhum teste de fórmula pega esse defeito, porque a fórmula
está certa; o que está errado é o texto que chegou. A recusa é a única
diferença observável entre "funcionou" e "cortou".

**Medição que sustenta.** Contra o servidor, o mesmo texto de 46 mil caracteres
devolve 400 com "the input length exceeds the context length" quando
`truncate` é false, e 200 com um vetor de 1.024 dimensões quando é true.

**O que isso obriga.** O teste marcado `gpu` em `tests/test_vetores.py` vetoriza
um texto longo, troca a última frase e confere que o vetor muda. Se o fim
estivesse sendo descartado, os dois sairiam iguais.
