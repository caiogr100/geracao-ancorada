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
de 4.280, todos tabela; o porquê está em `DECISOES.md`.
`python -m geracao_ancorada.estante conferir` carrega o que está em disco e se
recusa se a matriz, o registro e a assinatura divergirem, ou se o modelo no ar
não for o que construiu o índice.

## Dívidas declaradas

A tabela entra no índice sem a legenda que a nomeia, porque o fatiador separa
as duas e `estante/legenda.py` só recupera a legenda que estiver na mesma seção
e antes da tabela. Das 1.167 tabelas do índice, 115 recebem a legenda de volta.
O conserto definitivo pertence ao fatiador; o número fica registrado aqui para
não aparecer depois como surpresa.
