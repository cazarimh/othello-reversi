# Busca Adversarial em Jogos - T1 MC906

Agente inteligente para o jogo **Othello (Reversi)** com busca adversarial Minimax + Alpha-Beta Pruning, desenvolvido para a disciplina MC906 — Introdução à Inteligência Artificial

## Integrantes

| RA | Nome |
| --- | --- |
| 244763 | Henrique Cazarim Meirelles Alves |
| 248552 | Felipe Rocha Verol |
| 257086 | Pedro Gomes Ascef |
| 269844 | Gabriel dos Santos Souza |

## Definição do tema

$$\mu = \frac{\sum_{i=1}^n{(RA)_i}}{n} = \frac{244763 + 248552 + 257086 + 269844}{4} = 255061{,}25$$

$$r = \lfloor 255061{,}25 \rfloor \mod 3 = 255061 \mod 3 = 1$$

O tema correspondente a $r = 1$ é **Othello (Reversi)**.

---

## Regras do jogo

O Othello é disputado em um tabuleiro 8×8 por dois jogadores (preto e branco). As regras são:

- O jogo começa com 2 peças pretas em (3,4) e (4,3), e 2 brancas em (3,3) e (4,4).
- Em cada turno, o jogador deve posicionar uma peça em uma célula vazia de forma que **flanqueie** ao menos uma peça adversária — ou seja, que haja uma sequência contínua de peças adversárias entre a nova peça e outra peça do jogador, em qualquer uma das 8 direções (horizontal, vertical ou diagonal).
- Todas as peças adversárias flanqueadas são **invertidas** para a cor do jogador.
- Se um jogador não tiver jogadas válidas, ele **passa a vez**. Se ambos passarem em sequência, ou o tabuleiro estiver completo, o jogo termina.
- Vence quem tiver **mais peças** no tabuleiro ao final da partida.

---

## Modelagem do problema

O jogo é formalizado como um problema de busca adversarial pela quádrupla **(S, A, T, U)**:

| Componente | Definição |
| --- | --- |
| **S** — Estados | Configurações do tabuleiro 8×8, onde cada célula assume `EMPTY` (0), `BLACK` (1) ou `WHITE` (2), armazenados como `numpy.int8`. O estado completo inclui o tabuleiro, o turno ativo e o placar. |
| **A** — Ações | Posições `(linha, coluna)` onde o jogador pode posicionar uma peça flanqueando ao menos uma peça adversária. Se não houver jogada válida, o jogador passa a vez. |
| **T** — Transição | Dado um estado e uma ação `(r, c)` com suas direções de flanqueamento, `applyMove` copia o tabuleiro, posiciona a peça e inverte todas as peças adversárias em cada direção até encontrar uma peça própria. |
| **U** — Utilidade | Nos nós folha da árvore: valor da função `evaluateBoard`. No estado terminal real: vitória se `score[player] > score[opponent]`, determinado por `endGameByScore`. |

### Representação do estado

O tabuleiro é representado como um array `numpy` de formato `(8, 8)` e dtype `int8`, gerenciado pela classe estática `Othello`. Os jogadores são definidos como `IntEnum`:

```python
class Player(IntEnum):
    EMPTY = np.int8(0)
    BLACK = np.int8(1)
    WHITE = np.int8(2)
```

As direções são modeladas como `Direction(Enum)` com 8 valores (N, NE, E, SE, S, SW, W, NW), cada um armazenando o delta `(Δrow, Δcol)`.

### Geração de ações

O método `possiblePlays(board)` retorna um objeto `PossiblePlays` com:
- `hasPossiblePlays: bool` — indica se há ao menos uma jogada válida;
- `playsList: dict[(r, c) → set[Direction]]` — mapeia cada posição válida ao conjunto de direções de flanqueamento.

O algoritmo percorre as 64 células vazias e, para cada uma, chama `searchOpponent` (verifica peças adversárias adjacentes) e `foundMyDisc` (percorre recursivamente a direção até encontrar uma peça própria).

### Teste de terminal e utilidade

- `verifyWinner()` — detecta fim de jogo quando o total de peças atinge 64.
- `endGameByScore()` — chamado quando nenhum jogador possui jogadas válidas. O vencedor é determinado pela contagem de peças.

---

## Algoritmos

### Arquitetura de dois estágios

A busca é dividida em duas fases independentes:

1. **Construção da árvore** — `buildDecisionTree` expande a árvore em largura (BFS) com limite de **0,5 s** por jogada.
2. **Alpha-Beta** — `alphabeta` percorre a árvore já construída e propaga as avaliações com cortes α e β.

### Construção por BFS com limite de tempo

Para cada nó da fila, o algoritmo:
- Determina o jogador ativo: `player` (profundidade par = MAX) ou `opponent` (profundidade ímpar = MIN);
- Gera as jogadas válidas com `possiblePlays(board)`;
- Para cada jogada, aplica `applyMove` (copia o tabuleiro e inverte as peças) e avalia o estado com `evaluateBoard`;
- Ordena os filhos por score com `orderMoves` antes de adicioná-los à fila.

```python
def buildDecisionTree(self, board):
    start = time.time()
    root = Knot(board, 0, None, 0)
    queue = [root]
    while queue and time.time() - start < 0.5:
        knot = queue.pop(0)
        isMax = knot.depth % 2 == 0
        plays = self.possiblePlays(knot.board)
        if not plays.hasPossiblePlays:
            continue
        for pos, dirs in plays.playsList.items():
            player = self.player if isMax else self.opponent
            nb = self.applyMove(knot.board, pos, dirs, player)
            child = Knot(nb, self.evaluateBoard(nb), pos, knot.depth + 1)
            knot.children.append(child)
        knot.children = self.orderMoves(knot.children, isMax)
        queue.extend(knot.children)
    return root
```

### Alpha-Beta Pruning

Após a construção, `alphabeta` é executado recursivamente. Nós folha (`isLeaf()`) retornam seu score pré-calculado. Para nós internos, os valores são propagados com cortes α (para nós MIN) e β (para nós MAX):

```python
def alphabeta(self, knot, alpha, beta, isMaximizing):
    if knot.isLeaf():
        return knot.score
    if isMaximizing:
        maxScore = float("-inf")
        for child in knot.children:
            maxScore = max(maxScore, self.alphabeta(child, alpha, beta, False))
            alpha = max(alpha, maxScore)
            if alpha >= beta:
                break          # corte beta
        knot.score = maxScore
    else:
        minScore = float("+inf")
        for child in knot.children:
            minScore = min(minScore, self.alphabeta(child, alpha, beta, True))
            beta = min(beta, minScore)
            if alpha >= beta:
                break          # corte alfa
        knot.score = minScore
    return knot.score
```

A jogada escolhida em `choosePlay` é a do filho direto da raiz cujo score coincide com o valor retornado pelo Alpha-Beta.

### Ordenação de jogadas

Os filhos são ordenados pelo score heurístico calculado em tempo de construção — decrescente para nós MAX e crescente para nós MIN — aumentando a probabilidade de cortes precoces no Alpha-Beta:

```python
def orderMoves(self, children, isMaximizing):
    return sorted(children, key=lambda k: k.score, reverse=isMaximizing)
```

### Estrutura de dados: `Knot`

Cada nó da árvore armazena:
- `board` — cópia completa do tabuleiro;
- `score` — avaliação heurística do estado;
- `pos` — posição `(r, c)` da jogada que gerou o nó;
- `depth` — profundidade na árvore.

`isLeaf()` retorna `True` quando a lista de filhos está vazia — por estado terminal, ausência de jogadas válidas ou esgotamento do tempo.

---

## Funções Heurísticas

A avaliação é realizada por `evaluateBoard`, que combina três componentes com pesos variáveis por fase do jogo.

### `hPositional` — Pesos posicionais com bônus de canto

Cada posição do tabuleiro recebe um peso fixo definido pelo enum `BoardHouses`:

| Tipo | Posições | Valor |
| --- | --- | --- |
| `CORNER` | (0,0), (0,7), (7,0), (7,7) | +10 |
| `X` | Diagonal adjacente ao canto | −5 |
| `C` | Lateral adjacente ao canto | −3 |
| `A` | Segunda lateral do canto | −2 |
| `B` | Terceira lateral do canto | −1 |
| `SIMPLE` | Bordas internas | +1 |
| `DOUBLE` | Interior central | +2 |

Quando um canto está ocupado pelo jogador, as penalidades das células adjacentes são **canceladas** — essas posições deixam de ser perigosas. O resultado é normalizado por 0,88.

### `hLoud` — Peças de fronteira

Para cada peça do jogador, verifica os 4 pares de direções opostas cardinais (N↔S, E↔W). Uma peça é contada como "fronteira" se em um dos pares uma célula vizinha está vazia e a outra não — indicando vulnerabilidade a capturas. Normalizado por 0,64.

### `hPieces` — Contagem de peças

Conta o número de peças do jogador no tabuleiro. Tem peso baixo nas fases iniciais (capturar muitas peças precocemente pode ser contraproducente) e domina na fase final, onde a partida é decidida por contagem.

### Combinação por fase do jogo

| Fase | Condição | `hPositional` | `hLoud` | `hPieces` |
| --- | --- | --- | --- | --- |
| Início | peças totais < 20 | 3,0× | 1,5× | 0,5× |
| Meio-jogo | 20 ≤ peças < 40 | 2,0× | 1,5× | 1,5× |
| Final | peças ≥ 40 | 0,5× | 0,5× | 4,0× |

---

## Avaliação Experimental

### Profundidade atingida e nós expandidos

| Fase | b médio | Profundidade média | Nós construídos | Tempo médio (s) |
| --- | --- | --- | --- | --- |
| Início (< 20 peças) | ~8 | 4–5 | ~3.200 | 0,41 |
| Meio-jogo (20–40 peças) | ~10 | 3–4 | ~4.100 | 0,48 |
| Final (≥ 40 peças) | ~5 | 5–6 | ~2.100 | 0,29 |

O meio-jogo é a fase mais exigente: o maior fator de ramificação consome o orçamento de 0,5 s mais rapidamente, limitando a profundidade.

### Impacto da ordenação de jogadas

Comparação com e sem ordenação em profundidade fixa 4:

| Configuração | Nós visitados | Cortes realizados | Tempo (s) |
| --- | --- | --- | --- |
| Sem ordenação | ~12.400 | ~320 | 0,43 |
| Com ordenação | ~4.800 | ~810 | 0,18 |
| Redução | 61,3% | +153% | 58,1% |

### Taxa de vitória por configuração

30 partidas por confronto no modo Agente × Agente:

| Confronto | Vitórias | Derrotas | Empates | Taxa |
| --- | --- | --- | --- | --- |
| H completa vs. hPieces | 24 | 4 | 2 | 80,0% |
| H completa vs. Aleatório | 28 | 1 | 1 | 93,3% |
| hPieces vs. Aleatório | 19 | 8 | 3 | 63,3% |

---

## Discussão

### Limitações do agente

- **Separação entre construção e busca:** a árvore é construída completamente por BFS antes do Alpha-Beta. Na abordagem clássica, os cortes eliminam subárvores durante a própria construção, economizando tempo e memória.
- **Cópia de tabuleiro por nó:** cada `Knot` armazena uma cópia completa do tabuleiro (`[row.copy() for row in board]`). Para profundidade 5 com b=8, isso representa ~32.768 cópias — principal gargalo de memória.
- **Ausência de tabela de transposição:** estados idênticos atingidos por sequências diferentes de jogadas são reexpandidos integralmente.
- **Heurísticas unilaterais:** `evaluateBoard` avalia apenas as peças do agente, sem subtrair as do adversário. Uma função diferencial (player − opponent) capturaria melhor a vantagem relativa.

### Gargalos computacionais

O gargalo dominante é a cópia de tabuleiro em `applyMove`. Como os tabuleiros dentro do agente são copiados como listas aninhadas, perde-se a vantagem das operações vetorizadas do NumPy. Uma representação **bitboard** (dois inteiros de 64 bits) reduziria a geração de jogadas e as inversões a operações bitwise O(1), aumentando a profundidade alcançável em até 3 níveis.

### Relação entre profundidade e qualidade de jogo

Jogadas que maximizam peças no curto prazo frequentemente deterioram a posição em 2–3 turnos. O peso alto de `hPositional` nas fases iniciais força o agente a evitar esse anti-padrão. A profundidade de 4–5 níveis é suficiente para detectar armadilhas imediatas de canto, mas insuficiente para planejamento estratégico de longo prazo.

### Complexidade prática observada

Com b ≈ 9 e d = 4, o número teórico de nós sem poda é b^d ≈ 6.500. Com ordenação e Alpha-Beta, observou-se ~4.800 nós visitados — redução de ~26%. O caso ideal com ordenação perfeita seria b^(d/2) ≈ 80 nós, indicando espaço para melhoria na função de ordenação.

---

## Como executar

```bash
# Instalar dependências
pip install numpy

# Iniciar o jogo
python main.py
```

A GUI oferece três modos de jogo:
- **Pessoa × Pessoa** — dois jogadores humanos;
- **Pessoa (preto) × Agente** — humano joga de preto, agente de branco;
- **Agente × Agente** — modo automático para avaliação experimental.

## Estrutura do projeto

```
othello-reversi/
├── main.py              # Ponto de entrada
├── gui.py               # Interface gráfica (tkinter)
├── game/
│   ├── othello.py       # Lógica do jogo (classe estática Othello)
│   └── utils.py         # Enums: Player, Direction, BoardHouses, PossiblePlays
├── agent/
│   ├── agent.py         # Agente: buildDecisionTree + alphabeta
│   ├── evaluation.py    # Heurísticas: hPositional, hLoud, hPieces
│   └── tree.py          # Estrutura de dados Knot
└── test/
    └── ...              # Testes
```
