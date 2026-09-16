# Geração ancorada

Harness experimental do Trabalho de Conclusão de Curso "Geração ancorada de
comentários de questões de residência médica com modelos de linguagem
abertos executados localmente" (Ciência da Computação, UFPel, 2026/2 a
2027/1).

O trabalho avalia, por ablação pareada, quanto a recuperação em fontes
normativas brasileiras e o roteamento por tipo de questão contribuem para a
qualidade e para o uso verificável da fonte em comentários gerados por
modelos de pesos abertos rodando em hardware local. A proposta completa, com
o desenho experimental e o plano de análise, entra neste repositório quando
for aprovada.

## Estado

Em implantação. A proposta está em avaliação; o corpus, o índice de fontes e
a pipeline serão construídos aqui ao longo do TCC1 (setembro a dezembro de
2026), com todo o histórico público desde o início.

## Autoria

Caio Garcez Ribeiro (autor). Orientadora: profa. Larissa Astrogildo de
Freitas. Co-orientador: prof. Ulisses Brisolara Corrêa. O desenvolvimento do
código conta com assistência de ferramentas de IA, declarada na monografia;
os instrumentos de medição e as decisões de desenho são responsabilidade do
autor e dos orientadores.

A licença do código e a dos dados serão definidas junto com o pré-registro
do plano de análise.

## Índice

`python -m geracao_ancorada.estante indexar` reconstrói o índice inteiro a
partir do manifesto e grava dois destinos. O pesado, com o texto de cada pedaço
e a matriz de vetores, fica em `fontes/cache/estante/`, fora do repositório. O
leve fica em `indice/` e é versionado: `registro.json` traz uma linha de
etiquetas por pedaço (fonte, parte, seção, páginas, tipo, hash do texto) e
`assinatura.json` diz com o que o índice foi construído (modelo e digest do
Ollama, parâmetros do BM25, versão do tokenizador, commit do fatiador) e o que
ficou de fora, com o motivo. Nenhum dos dois contém texto de fonte.

O servidor de embedding recusa o pedaço que passa do teto de contexto em vez
de cortá-lo em silêncio, e o pedaço recusado fica fora da matriz e dentro da
assinatura, identificado pelo hash do texto. Na versão atual do corpus são 10
de 4.298, todos tabela; o porquê está em `DECISOES.md`.
`python -m geracao_ancorada.estante conferir` carrega o que está em disco e se
recusa se a matriz, o registro e a assinatura divergirem, ou se o modelo no ar
não for o que construiu o índice.

## Busca

`python -m geracao_ancorada.estante buscar "pergunta"` roda os dois braços
sobre o índice carregado e entrega oito pedaços. O braço léxico é o BM25 do
índice, sobre os tokens da pergunta menos as palavras funcionais; o denso é o
produto interno do vetor da pergunta com a matriz. Cada braço traz os seus
cinquenta primeiros, e a lista léxica para no último pedaço com escore acima
de zero. As duas listas se fundem por posição (RRF, constante 60), e o primeiro
colocado de cada braço entra sempre na entrega, na posição que a fusão lhe dá.
O empate se resolve pelo id.
As quatro constantes ficam gravadas na assinatura do índice, e a decisão, com
a aritmética que a sustenta, está em `DECISOES.md`.

Cada achado carrega a posição em cada lista, o cosseno e o BM25 crus, o escore
fundido e a marca de ter entrado pela garantia. `--saida corridas.jsonl`
acrescenta a corrida a um arquivo com todos os candidatos e os dois escores,
sem o texto, o que permite recalcular a fusão com outra constante ou sem a
garantia sem rodar a busca de novo. `--area` e `--fonte` filtram por máscara
antes da pontuação; a área casa por interseção, porque três documentos têm
duas. O pedaço descartável (bibliografia, sumário) fica de fora por padrão.

## Dívidas declaradas

A tabela entra no índice sem a legenda que a nomeia, porque o fatiador separa
as duas e `estante/legenda.py` só recupera a legenda que estiver na mesma seção
e antes da tabela. Das 1.164 tabelas do índice, 72 recebem a legenda de volta;
eram 115 antes de a legenda de figura deixar de valer para tabela, e 55 dessas
eram fluxogramas batizando tabela de critérios. O conserto definitivo pertence
ao fatiador; o número fica registrado aqui para não aparecer depois como
surpresa.
