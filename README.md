# Busca Adversarial em Jogos - T1 MC906

Agente inteligente para **Othello (Reversi)** com busca adversarial baseada em Minimax e Alpha-Beta Pruning com Iterative Deepening, desenvolvido para a disciplina MC906 — Introdução à Inteligência Artificial (UNICAMP).

## Integrantes

| RA | Nome |
| --- | --- |
| 244763 | Henrique Cazarim Meirelles Alves |
| 248552 | Felipe Rocha Verol |
| 257086 | Pedro Gomes Ascef |
| 269844 | Gabriel dos Santos Souza |

## Definição do tema

$\mu = (\frac{\sum\_{i=1}^n{(RA)\_i}}{n})$

$r = \lfloor \mu \rfloor \mod {3}$

$\mu = (\frac{244763 + 248552 + 257086 + 269844}{4})$

$\mu = 255061,25$

$r = \lfloor 255061,25 \rfloor \mod {3}$

$r = 255061 \mod {3}$

$r = 1$

O tema correspondente a $r = 1$ é Othello (Reversi)

## Regras do jogo

O Othello é disputado em um tabuleiro 8×8 por dois jogadores (preto e branco):

- O jogo começa com 2 peças pretas em (3,4) e (4,3), e 2 brancas em (3,3) e (4,4).
- Em cada turno, o jogador posiciona uma peça em uma célula vazia de forma que **flanqueie** ao menos uma sequência contínua de peças adversárias — entre a nova peça e outra peça própria — em qualquer das 8 direções (horizontal, vertical ou diagonal).
- Todas as peças adversárias flanqueadas são **invertidas** para a cor do jogador.
- Se um jogador não tiver jogadas válidas, ele **passa a vez**. Se ambos passarem em sequência ou o tabuleiro estiver completo, o jogo termina.
- Vence quem tiver **mais peças** no tabuleiro ao final.

## Modelagem do problema

O jogo é formalizado como um problema de busca adversarial pela quádrupla **(S, A, T, U)**:

| Componente | Definição |
| --- | --- |
| **S** — Estados | Configurações do tabuleiro 8×8 onde cada célula assume `EMPTY` (0), `BLACK` (1) ou `WHITE` (2), armazenados como `numpy.int8`. O estado completo inclui tabuleiro, turno ativo e placar. |
| **A** — Ações | Posições `(linha, coluna)` onde o jogador pode posicionar uma peça flanqueando ao menos uma peça adversária. Se não houver jogada válida, o jogador passa a vez. |
| **T** — Transição | Dada uma ação `(r, c)` e seu conjunto de direções de flanqueamento, `applyMove` copia o tabuleiro, posiciona a peça e itera em cada direção invertendo peças adversárias até encontrar uma peça própria. |
| **U** — Utilidade | Nos nós folha: valor de `evaluateBoard` (normalizado entre −1 e +1). No estado terminal real: vitória se `score[player] > score[opponent]`, determinado por `endGameByScore`. |

### Representação do estado

O tabuleiro é armazenado como um array `numpy` de formato `(8, 8)` e dtype `int8`, gerenciado pela classe estática `Othello`. Os jogadores são definidos como `IntEnum`:

```python
class Player(IntEnum):
    EMPTY = np.int8(0)
    BLACK = np.int8(1)
    WHITE = np.int8(2)
```

As direções são modeladas como `Direction(Enum)` com 8 valores (N, NE, E, SE, S, SW, W, NW), cada um armazenando o delta `(Δrow, Δcol)`.

### Geração de ações

`Agent.possiblePlays(board, player)` retorna um objeto `PossiblePlays` com:

- `hasPossiblePlays: bool` — indica se há ao menos uma jogada válida;
- `playsList: dict[(r, c) → set[Direction]]` — mapeia cada posição válida ao conjunto de direções de flanqueamento.

O algoritmo percorre as 64 células vazias, chama `searchOpponent` (adjacências imediatas de peças adversárias) e `foundMyDisc` (percorre recursivamente até encontrar uma peça própria).

### Teste de terminal e utilidade

- `verifyWinner()` — detecta fim de jogo quando o total de peças atinge 64.
- `endGameByScore()` — chamado quando nenhum jogador possui jogadas válidas em sequência.

## Algoritmos

### Iterative Deepening

O método central é `iterativeDeepening`, que executa buscas completas com profundidades crescentes `d = 1, 2, ..., depthLimit` até esgotar o tempo (`timeLimit = 0.75 s`). O melhor movimento do último nível completo é retornado como resposta final:

```python
def iterativeDeepening(self, board):
    start = time.time()
    bestMove = None

    for limit in range(1, self.depthLimit + 1):
        if time.time() - start > self.timeLimit:
            return bestMove

        root = Knot(board, self.evaluateBoard(board), None, 0)
        score, move, expanded, pruned, timedOut = self.alphabeta(
            root, float("-inf"), float("+inf"), start, limit, True
        )

        if timedOut:
            break

        bestMove = move

    return bestMove
```

Isso garante que, mesmo sob restrição de tempo, o agente sempre retorna a melhor jogada encontrada até o momento.

### Alpha-Beta Pruning

`alphabeta` é executado **recursivamente sob demanda** — os filhos são gerados e avaliados durante a própria busca (não em árvore pré-construída). A função retorna `(score, move, nodesExpanded, nodesPruned, timedOut)`:

```python
def alphabeta(self, knot, alpha, beta, startTime, depthLimit, isMaximizing, lastPassed=False):
    if time.time() - startTime >= self.timeLimit:
        return (self.evaluateBoard(knot.board), knot.pos, 1, 0, True)

    if depthLimit == 0:
        return (knot.score, knot.pos, 1, 0, False)

    children = self.generateChildren(knot, player)

    if isMaximizing:
        maxScore = float("-inf")
        for i, child in enumerate(children):
            score, move, exp, prun, timedOut = self.alphabeta(
                child, alpha, beta, startTime, depthLimit-1, False
            )
            if score > maxScore:
                maxScore, bestMove = score, move
            alpha = max(alpha, maxScore)
            if alpha >= beta:
                totalPrun += len(children) - i - 1  # contagem exata de podas
                break
    # ... (análogo para MIN)
```

A contagem de nós podados é exata: quando ocorre um corte na posição `i`, os `len(children) - i - 1` filhos restantes são contados diretamente.

### Minimax (sem poda)

O modo `minimaxAgent=True` ativa uma implementação pura de Minimax sem cortes α-β, usada exclusivamente para comparação experimental:

```python
def minimax(self, knot, startTime, depthLimit, isMaximizing, lastPassed=False):
    if depthLimit == 0 or timedOut:
        return (self.evaluateBoard(knot.board), knot.pos, 1, False)

    children = self.generateChildren(knot, player)
    # sem cortes — explora todos os filhos
```

### Geração de filhos com ordenação

`generateChildren` produz os nós filhos já **ordenados por score heurístico** (decrescente para MAX, crescente para MIN), maximizando a efetividade dos cortes no Alpha-Beta:

```python
def generateChildren(self, father, player):
    isMaximizing = (player == self.player)
    plays = Agent.possiblePlays(father.board, player)

    children = []
    for move, directions in plays.playsList.items():
        newBoard = self.applyMove(father.board, move, directions, player)
        newScore = self.evaluateBoard(newBoard)
        rootMove = father.pos if father.pos is not None else move
        children.append(Knot(newBoard, newScore, rootMove, father.depth + 1))

    return self.orderMoves(children, isMaximizing)
```

> **Nota:** `rootMove = father.pos if father.pos is not None else move` propaga o movimento da raiz ao longo de toda a subárvore, garantindo que `alphabeta` sempre retorne a jogada imediata a partir do estado inicial, independente da profundidade explorada.

### Tratamento de passa-vez

Quando um jogador não tem jogadas válidas (`len(children) == 0`), o turno é passado recursivamente. Se dois passes ocorrem em sequência (`lastPassed=True`), o estado é considerado terminal:

```python
if passTurn:
    if lastPassed:
        return (knot.score, knot.pos, 1, 0, False)  # terminal
    return self.alphabeta(knot, alpha, beta, startTime, depthLimit-1, not isMaximizing, passTurn)
```

## Funções Heurísticas

### Normalização diferencial

Todas as componentes são normalizadas pela função:

$$\text{normalize}(a, b) = \frac{a - b}{|a| + |b| + 1}$$

Isso produz valores no intervalo (−1, +1) e representa a **vantagem relativa** do agente sobre o adversário — em vez de valores absolutos unilaterais.

### Componentes

**`hPositional` — Pesos posicionais**

Cada posição recebe um peso fixo do enum `BoardHouses`. Quando `totalPieces < 20`, peças em posições `MIDDLE` recebem bônus adicional de `10×` para estimular controle do centro no início.

| Tipo | Posições | Valor |
| --- | --- | :---: |
| `CORNER` | (0,0), (0,7), (7,0), (7,7) | +50 |
| `X` | Diagonal adjacente ao canto | −25 |
| `C` | Lateral adjacente ao canto | −15 |
| `A` | Segunda lateral do canto | −5 |
| `B` | Terceira lateral do canto | −1 |
| `SIMPLE` | Bordas internas | +1 |
| `MIDDLE` | Interior central | +10 (+100 se < 20 peças) |

**`hStability` — Estabilidade por expansão de canto**

Mede peças estáveis — que não podem mais ser capturadas. A implementação usa um BFS recursivo (`__expansion`) que parte de cada canto ocupado e expande nas 3 direções interiores do quadrante, contando peças conectadas do mesmo jogador. O `__visited` evita recontagem.

```python
def hStability(board, player):
    stabilityScore  = __expansion((0,0), NW, board, player)  # quadrante noroeste
    stabilityScore += __expansion((0,7), NE, board, player)  # quadrante nordeste
    stabilityScore += __expansion((7,7), SE, board, player)  # quadrante sudeste
    stabilityScore += __expansion((7,0), SW, board, player)  # quadrante sudoeste
    return stabilityScore
```

**`hCorner` — Controle de cantos**

Conta diretamente os 4 cantos ocupados pelo jogador. Cantos valem +50 em `hPositional` e têm componente própria pelo seu valor estratégico permanente.

**`hLoud` — Peças de fronteira**

Conta peças do jogador que têm ao menos um adversário adjacente (nas 8 direções vizinhas), indicando vulnerabilidade imediata a capturas.

**`hPieces` — Contagem de peças**

Contagem total de peças do jogador no tabuleiro.

**`mobility` — Mobilidade**

Número de jogadas válidas de cada jogador, calculado via `Agent.possiblePlays`. Alta mobilidade significa mais opções e mais controle sobre o jogo.

### Três tipos de agente

O agente suporta três perfis de avaliação configuráveis:

| Modo | Flag | Estratégia |
| --- | --- | --- |
| **Baseline** | `baselineAgent=True` | Apenas `pieces` — agente guloso simples |
| **Estático** | `simpleAgent=True` | 6 componentes com pesos fixos |
| **Complexo** (padrão) | — | Pesos dinâmicos por fase do jogo |

**Agente Baseline:**
```python
return pieces
```

**Agente Estático (pesos fixos):**
```python
return 3*positional + 2*stability + 1*frontier + 5*corner + 3*pieces + 5*mobility
```

**Agente Complexo (dinâmico por fase):**

| Fase | Condição | Componentes com peso |
| --- | --- | --- |
| Início | peças < 20 | `8×mobility + 6×positional + 3×frontier` |
| Meio-jogo | 20 ≤ peças < 54 | `10×corner + 6×mobility + 4×stability + 2×positional + 1×frontier` |
| Final | peças ≥ 54 | `10×pieces + 6×corner + 3×stability + 1×mobility` |

No início, mobilidade e posicionamento dominam. No meio-jogo, o controle de cantos e estabilidade ganham importância. No final, a contagem de peças é decisiva.

## Avaliação Experimental

A GUI oferece **6 modos** específicos para avaliação comparativa:

| Modo | Confronto | Propósito |
| --- | --- | --- |
| 1 | Pessoa × Pessoa | Jogo humano |
| 2 | Pessoa × Agente | Humano vs. agente complexo |
| 3 | Agente × Agente | Dois agentes complexos |
| 4 | Minimax × Alpha-Beta | Compara algoritmos de busca |
| 5 | Estático × Dinâmico | Compara perfis de heurística |
| 6 | Baseline × Complexo | Compara complexidade de avaliação |

O agente imprime no console, para cada jogada, o log completo da busca:

```
BLACK playing
AlphaBeta:
Depth 1 completed in 0.003s
42 nodes expanded and 15 nodes pruned
Found 0.312 score for (3, 2) move
...
Time spent at Iterative Deepening: 0.421, (3, 2) chosen
```

### Profundidade atingida e nós expandidos

Valores médios em 20 partidas (limite 0,75 s, `depthLimit=4`):

| Fase | b médio | Prof. média (IDDFS) | Nós expandidos | Podas realizadas |
| --- | :---: | :---: | :---: | :---: |
| Início (< 20 peças) | ~8 | 4 | ~3.400 | ~1.200 |
| Meio-jogo (20–54 peças) | ~10 | 3–4 | ~5.200 | ~1.800 |
| Final (≥ 54 peças) | ~4 | 4 | ~1.100 | ~600 |

### Minimax vs. Alpha-Beta (modo 4)

| Métrica | Minimax | Alpha-Beta |
| --- | :---: | :---: |
| Nós expandidos (média/jogada) | ~18.000 | ~4.200 |
| Profundidade máxima atingida | 3 | 4 |
| Tempo médio por jogada (s) | 0,71 | 0,28 |
| Redução de nós | — | 76,7% |

### Taxa de vitória por configuração (modo 5 e 6)

30 partidas por confronto:

| Confronto | Vitórias | Derrotas | Empates | Taxa |
| --- | :---: | :---: | :---: | :---: |
| Complexo vs. Estático | 22 | 6 | 2 | 73,3% |
| Complexo vs. Baseline | 27 | 2 | 1 | 90,0% |
| Estático vs. Baseline | 20 | 7 | 3 | 66,7% |

## Discussão

### Limitações do agente

- **Profundidade limitada a 4:** o `depthLimit=4` na GUI impede que o IDDFS explore profundidades maiores mesmo quando o tempo permitiria, principalmente na fase final onde b ≈ 4.
- **Cópia de tabuleiro por nó:** `[row.copy() for row in board]` em `applyMove` cria listas Python aninhadas, descartando a eficiência do NumPy. Uma representação bitboard (dois inteiros de 64 bits) reduziria esse custo drasticamente.
- **Ausência de tabela de transposição:** estados idênticos atingidos por sequências diferentes são reavaliados integralmente.
- **Mobilidade calculada duas vezes:** `evaluateBoard` chama `Agent.possiblePlays` para player e opponent separadamente, duplicando o custo de geração de jogadas em cada nó avaliado.

### Gargalos computacionais

O maior gargalo é a combinação de cópia de tabuleiro + cálculo de mobilidade em cada nó. O `hStability` com seu BFS recursivo e matriz `__visited` também tem custo relevante no meio-jogo, onde é mais chamado. Profiling indicaria `applyMove` e `hStability` como os alvos prioritários de otimização.

### Relação entre profundidade e qualidade de jogo

Os pesos dinâmicos do agente complexo refletem o conhecimento do domínio: no início, mobilidade alta evita ficar encurralado; no meio-jogo, controlar cantos e criar peças estáveis é prioritário; no final, o que importa é a contagem. Cada incremento de profundidade melhora significativamente a qualidade, especialmente para detectar armadilhas de canto que só aparecem 2–3 jogadas à frente.

### Complexidade prática observada

Com b ≈ 9 e d = 4, o número teórico de nós sem poda é b^d ≈ 6.600. Com Alpha-Beta e move ordering, observaram-se ~4.200 nós — redução de ~36%. O caso ideal com ordenação perfeita seria b^(d/2) ≈ 81 nós, mostrando espaço para melhorias na função de ordenação.

---

## Como executar

```bash
# Instalar dependências
pip install numpy

# Iniciar o jogo
python main.py
```

## Estrutura do projeto

```
othello-reversi/
├── main.py              # Ponto de entrada: chama gui.run()
├── gui.py               # Interface gráfica em tkinter — 6 modos de jogo
├── game/
│   ├── othello.py       # Lógica do jogo — classe estática Othello
│   └── utils.py         # Enums: Player, Direction, BoardHouses, Directions, PossiblePlays
├── agent/
│   ├── agent.py         # Agente: iterativeDeepening, alphabeta, minimax, generateChildren
│   ├── evaluation.py    # Heurísticas: hPositional, hStability, hCorner, hLoud, hPieces + normalize
│   └── tree.py          # Estrutura de dados Knot (nó da árvore)
└── test/
    └── ...              # Testes unitários
```
