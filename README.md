# Busca Adversarial em Jogos - T1 MC906

Agente inteligente para Othello (Reversi) com busca adversarial baseada em Minimax e Alpha-Beta Pruning com Iterative Deepening, desenvolvido para a disciplina MC906 — Introdução à Inteligência Artificial (UNICAMP).

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
- Em cada turno, o jogador posiciona uma peça em uma célula vazia de forma que flanqueie ao menos uma sequência contínua de peças adversárias, entre a nova peça e outra peça própria, em qualquer das 8 direções.
- Todas as peças adversárias flanqueadas são invertidas para a cor do jogador.
- Se um jogador não tiver jogadas válidas, ele passa a vez. Se ambos passarem em sequência ou o tabuleiro estiver completo, o jogo termina.
- Vence quem tiver mais peças no tabuleiro ao final.


## Como executar

```bash
# Clonar o repositório
# Instalar dependências

Debian/Ubuntu:
sudo apt update
sudo apt install python3-tk

Fedora:
sudo dnf update
sudo dnf install python3-tkinter

MacOS:
brew install tcl-tk

# Entrar no repositório do projeto
# Iniciar o jogo
python3 main.py
```

Com isso aparecerá a tela com o jogo, e um menu com todas as opções implementadas.

## Estrutura do projeto

```
othello-reversi/
├── main.py              # Ponto de entrada: chama gui.run()
├── .gitignore
├── README.md
├── gui/
    ├── gui.py                 # Interface gráfica em tkinter — 6 modos de jogo
├── game/
│   ├── othello.py       # Lógica do jogo — classe estática Othello
│   └── utils.py         # Enums: Player, Direction, BoardHouses, Directions, PossiblePlays
├── agent/
│   ├── agent.py         # Agente: iterativeDeepening, alphabeta, minimax, generateChildren
│   ├── evaluation.py    # Heurísticas: hPositional, hStability, hCorner, hLoud, hPieces + normalize
│   └── tree.py          # Estrutura de dados Knot (nó da árvore)
└── metrics/
    └── ...              # Métricas usadas para testar o agente
```
