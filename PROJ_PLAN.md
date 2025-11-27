Here’s how I’d turn your Onitama repo into a little RL lab, step-by-step.

You already have:

* A solid game engine (`src/game`), random + heuristic agents, CLI for running matches (`main.py`) and a PPO scaffold (`ppo_agent.py`, `train_ppo.py`). ([GitHub][1])

So the question is: what sequence of agents will teach you the most about RL, training, and eval, without getting lost in engineering?

---

## Big picture roadmap

I’d structure it into 6–7 “projects”, each centered around a new agent + 1–2 key RL concepts:

1. **Beefed-up heuristic + evaluation harness**
2. **Tabular RL on a simplified Onitama (Q-learning / SARSA)**
3. **Linear function approximation (feature-based Q / V)**
4. **First neural policy: REINFORCE with baseline**
5. **Full actor-critic + PPO (filling in your PPOAgent)**
6. **Search-based agent: MCTS for comparison**
7. **Advanced: self-play leagues, curriculum, and imitation**

Below I’ll go through each: intuition first, then a bit of math, plus concrete design decisions to make in your repo.

---

## 1. Stronger heuristic + evaluation harness

**Goal:** Tighten your classical baseline and build the tools you’ll reuse for all RL experiments.

### Agent

* Extend `HeuristicAgent` from “simple rules” to a **linear evaluation function**:

  * Features φ(s): material balance, king safety (master distance from capture zones), pawn advancement, mobility (#legal moves), control of central squares, etc.
  * Score:
    [
    V(s) = w^\top \phi(s)
    ]
  * Agent picks the legal move that maximizes V(resulting_state).

Even if you don’t learn w yet, designing φ(s) forces you to think like an RL value function.

### Infrastructure

* Add a **tournament / benchmarking helper**:

  * Function to run N games between any two agents with:

    * Randomized initial card sets.
    * Colors swapped halfway.
  * Return:

    * Win rate, draw rate.
    * Approximate **confidence interval** via binomial:
      [
      \hat p = \frac{\text{wins}}{N}, \quad
      \text{SE} = \sqrt{\frac{\hat p (1-\hat p)}{N}}
      ]
  * Optional: simple Elo update routine.

This gives you a standard way to say “this new RL agent is actually better than heuristic by X ± Y”.

---

## 2. Tabular RL on a simplified Onitama

**Goal:** Feel the core MDP machinery—Q(s, a), Bellman backup, exploration vs exploitation—without function approximation headaches.

### Simplify the game

Create `SimpleOnitamaEnv` (maybe as a separate “mode” in `game.py`):

* Fixed small card set (e.g. 3–5 cards total).
* No card swapping, or reduced mechanics.
* Possibly smaller board (3×3 or 4×4) if that’s easy.
* Terminate after a small horizon (e.g. 20 moves) if no win.

The key: **small enough state-action space to allow tabular learning**.

### RL algorithm: Q-learning or SARSA

* State representation:

  * Convert board + card indices + current player into a hashable key (e.g. tuple of ints) for a Python dict.

* Q-learning update:
  [
  Q(s,a) \leftarrow Q(s,a) + \alpha\Big(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big)
  ]
  where:

  * γ ≈ 0.99 (episodic game).
  * Reward r is 0 for all moves, +1 at win, −1 at loss.

* Behavior policy: ε-greedy over Q(s, a).

**What you’ll learn / decide:**

* How **state/action representations** affect feasibility.
* Trade-off between **γ** and how far credit propagates.
* Exploration schedules (ε decay, etc.).
* Stability vs convergence speed.

Once trained on the toy environment, evaluate this agent vs random and heuristic **on that same simplified game** using your tournament harness.

---

## 3. Linear function approximation on the full game

**Goal:** Move from tabular to **feature-based value functions** and see the bias/variance trade-off.

### Agent

* Reuse the features φ(s, a) you used in the improved heuristic (possibly extend them).
* Define:
  [
  Q_\mathbf{w}(s,a) = \mathbf{w}^\top \phi(s,a)
  ]
* Use **on-policy SARSA** or **TD(0)** with function approximation:

  * For SARSA:
    [
    \delta_t = r_t + \gamma Q_\mathbf{w}(s_{t+1}, a_{t+1}) - Q_\mathbf{w}(s_t, a_t)
    ]
    [
    \mathbf{w} \leftarrow \mathbf{w} + \alpha, \delta_t, \phi(s_t, a_t)
    ]

### Design decisions

* **Features**:

  * Material differences by piece type.
  * Master distance to own shrine / opponent shrine.
  * # legal moves available.
  * Threat counts (pieces that can be captured next move).
* **Onitama symmetry**:

  * Maybe encode states always from “current player’s perspective” (flip board, swap colors), so the same Q works for both sides.

**What you’ll learn:**

* Why off-policy + function approximation can blow up.
* How feature design affects learning vs your hand-tuned heuristic.
* How much learning you can squeeze out of **linear** models before going neural.

---

## 4. First neural policy: REINFORCE with baseline

**Goal:** Build an end-to-end neural policy for the real game and see classic policy-gradient behavior (high variance, need for baselines).

### State / action representation

* State tensor:

  * 5×5×C, where channels encode:

    * current player’s pieces (master/pawns),
    * opponent pieces,
    * shrines,
    * maybe card patterns (e.g., per-card embedding concatenated).
* List of legal moves from `game.py`. Map each legal (piece, card, move_offset) into a discrete action index in [0, K).

### Policy network

* πθ(a | s): small CNN or MLP → logits over all possible moves (mask illegal ones).
* Baseline Vψ(s): small value head (or separate network) predicting expected return.

### Algorithm: REINFORCE with baseline

For an episode:

* Collect (s_t, a_t, r_t). Compute returns:
  [
  G_t = \sum_{k=t}^T \gamma^{k-t} r_k
  ]

* Policy gradient update:
  [
  \nabla_\theta J \approx \mathbb{E}\big[(G_t - b_t)\nabla_\theta \log \pi_\theta(a_t|s_t)\big]
  ]
  where baseline b_t ≈ Vψ(s_t).

* Fit baseline by minimizing:
  [
  \min_\psi \mathbb{E}\big[(G_t - V_\psi(s_t))^2\big]
  ]

**What you’ll learn / tune:**

* Effect of **reward sparsity**: DO you need intermediate rewards (e.g. small reward for capture / piece loss)?
* Variance reduction via baseline; learning rate schedules.
* How long it takes to beat random / heuristic with pure policy gradient.

This can live as `PgAgent` + `train_pg.py` (or integrated into your PPO training script, sharing state encoding).

---

## 5. Actor-critic + PPO (complete your PPOAgent)

Now you’re ready to make the repo’s PPO scaffolding real.

### Shared actor-critic network

* Single network fθ(s) with:

  * Policy head: logits over actions → πθ(a|s).
  * Value head: Vθ(s).

### Advantage estimates (actor-critic)

* Temporal-difference error:
  [
  \delta_t = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)
  ]
* Use **GAE(λ)** if you want:
  [
  A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
  ]

### PPO objective

For each policy update:

* Old policy πθ_old fixed for the collected trajectories.
* Probability ratio:
  [
  r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
  ]
* Clipped loss:
  [
  L^{\text{clip}}(\theta) =
  \mathbb{E}\Big[
  \min\big(
  r_t(\theta) A_t,,
  \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
  \big)
  \Big]
  ]
* Total objective (maximize):
  [
  L(\theta) = \mathbb{E}[L^{\text{clip}}(\theta) + c_v L^{\text{value}}(\theta) + c_{\text{ent}} H[\pi_\theta(\cdot|s_t)]]
  ]
  where:

  * (L^{\text{value}}) is value MSE,
  * H is entropy for exploration.

Your existing `train_ppo.py` already hints at this flow. You’ll mainly fill in:

* State encoder.
* Actor-critic network.
* Trajectory buffer and GAE.
* PPO update loop (epochs, minibatches).

**Self-play design choices**

* Self-play vs fixed opponent:

  * Start training vs random or heuristic.
  * Then switch to self-play / periodic snapshot opponents.
* Symmetry:

  * Always encode states from current player’s perspective.
  * Use one shared network for both colors.

Evaluation: regularly pit the PPO agent against heuristic + your earlier agents; track Elo over training time.

---

## 6. Monte Carlo Tree Search (MCTS) agent

**Goal:** Compare RL to a **search-based agent** and understand why combining them (AlphaZero-style) works.

### Agent

* Implement a generic MCTS using your existing game API:

  * Node: (state, player_to_move).
  * Edges: legal moves with associated stats (N, W, Q, P).
* Pure MCTS variant:

  * Use UCB for selection.
* Advanced variant:

  * Use your value network (from actor-critic/PPO) as a **leaf evaluator** / prior (PUCT).

Then:

* Compare MCTS vs heuristic vs PPO.
* Play with **rollout depth**, number of simulations, and see how far pure search can go without huge networks.

---

## 7. Advanced topics: curriculum, imitation, evaluation

At this point, you have a whole zoo of agents. Now use them to explore higher-level RL topics:

### Curriculum learning

* Start training from easier distributions:

  * Late-game positions (few pieces).
  * Restricted card sets.
* Gradually increase difficulty: full card set, earlier in the game.

### Imitation + RL

* Collect games from your **strong heuristic** or MCTS as “expert” demonstrations.
* Do **behavior cloning**:

  * Cross-entropy loss between expert moves and πθ(a|s).
* Then fine-tune with PPO. You’ll see:

  * Much faster initial learning.
  * Trade-offs between copying expert errors vs exploring beyond them.

### Evaluation methodology

* Standardize:

  * Fixed test suite: say 10k games against each baseline (random, heuristic, MCTS with small budget).
  * Log win rates + Wilson CIs, maybe an Elo rating per agent.
* Optionally:

  * Track **sample efficiency**: win rate vs number of environment steps.

---

[1]: https://github.com/cghubbard/Onitama "GitHub - cghubbard/Onitama: RL Understand Work"


Nice, this is a great place to slow down and be intentional before coding.

You already have:

* **Random agent** ✅
* **Simple heuristic agent** ✅

Let’s dig into plans for:

* **B. “Real” heuristic agent** (refine + parameterize what you have)
* **C. 1-ply lookahead agent** (search on top of that heuristic)
* **D. Shallow minimax agent** (adversarial search with that heuristic)

I’ll keep it intuitive first, then sprinkle in some math.

---

## B. Heuristic Agent – what to design / experiment with

### 1. Intuition: what’s the job of this agent?

This agent is your **hand-crafted “value function”**: given a board position (s), it returns a number (V(s)) that means “how good this is for the current player.”

Later you can:

* Use it directly as a baseline agent.
* Use it as a **feature template** or **initialization** for RL value functions.
* Use it as a scoring function inside 1-ply and minimax agents.

So Step B is really “design a reasonable evaluation function for Onitama.”

---

### 2. Structure: linear heuristic is enough (and nice for learning later)

A natural structure:

[
V(s) = \sum_{k} w_k , \phi_k(s)
]

* (\phi_k(s)): hand-crafted features of state (s).
* (w_k): weights you choose (or later learn).

You can start with **manually chosen** (w_k), then later:

* Fit them with least squares or logistic regression using self-play data.
* Or learn them via RL.

---

### 3. Candidate features (\phi_k(s)) for Onitama

You don’t need all of these, but think in buckets:

**(1) Material & king safety**

* `#my_students - #opp_students`
* `my_master_alive` / `opp_master_alive`
* Big constant for “I win / I lose”:

  * `win_flag` (master capture or temple reach)
  * `lose_flag`

**(2) Mobility & threats**

* `#legal_moves_current_player`
* `#my_moves_that_capture_any_piece`
* `#opp_moves_that_capture_my_master` (king-in-check type feature)
* `#my_moves_that_threaten_master_next_turn` (how many ways I can threaten their master)

**(3) Positional / progress features**

* Average distance of my pieces to opponent temple.
* Minimum distance of my master to opponent temple.
* “Centralization”: count of my pieces in central 3×3 vs edge squares.
* Symmetry features (e.g., reflect board so you always view from current player’s perspective).

**(4) Card-related features**
This is Onitama-specific and powerful:

* For each card I hold:

  * Number of legal moves given this card.
  * Whether card has forward/backward movement.
  * Whether the card can **ever** reach opponent temple if my master walks that pattern repeatedly.
* For each opponent card (known information):

  * Number of moves they have that could capture my master.
  * Whether they have a “long” pattern vs only short steps.

These allow your heuristic to “understand” that having certain cards is really valuable.

---

### 4. Design decisions & experiments for B

Things you can explicitly play with:

1. **Feature set ablations**

   * Start minimalist: material + terminal + mobility.
   * Add card-awareness.
   * Add king-safety and see how it shifts style (more cautious vs reckless).

2. **Aggressiveness parameter**

   * Have a knob (\alpha) that scales capture-related features:
     [
     V_\alpha(s) = w^\top \phi(s) + \alpha \cdot (\text{capture_threat_features})
     ]
   * This lets you create “personalities”: more aggressive vs more positional.

3. **Perspective**

   * Decide whether (V(s)) is always from **current player’s viewpoint** or always from some fixed color’s viewpoint.
   * My suggestion: **always from the current player** for 1-ply and minimax; makes the logic less error-prone.

4. **Smoothing vs sharp thresholds**

   * Rather than “if my master is in danger add -100”, spread danger penalties more smoothly based on number of attacking moves or distance.

5. **Manual vs data-driven weights (w_k)**

   * Start with your best guess.
   * Later:

     * Log game states and outcomes from self-play.
     * Fit (w) with simple regression to approximate win probability.

**Outcome of Step B:**
You have a **clear, parametric heuristic** (V(s; w)) with a list of features and design knobs you can tweak.

---

## C. 1-Ply Lookahead Agent – wrapping search around the heuristic

### 1. Intuition

Your heuristic answers: “How good is this board right now?”

1-ply lookahead does:

> “For each legal move, imagine I play it, score the resulting board with the heuristic, and pick the best.”

So the policy is:

[
a^* = \arg\max_{a \in \mathcal{A}(s)} V(s_a)
]

where (s_a) is the state after applying action (a).

This is still **greedy**, but it anticipates immediate tactical gains or blunders that the pure heuristic might miss when only looking at the current board.

---

### 2. Core design decisions

1. **How to handle terminal states**

   * If a move results in immediate win or loss, short-circuit:

     * Assign (+\infty) or a very large constant for wins.
     * Assign (-\infty) or very negative constant for self-mate.

2. **Whose perspective is V?**

   * If (V) is always from current player’s perspective:

     * Just evaluate (V(s_a)) directly; no sign flipping.
   * If from a fixed color, you MUST flip signs depending on who’s about to play. I’d avoid this and stick to “current player perspective.”

3. **Tie-breaking**
   When multiple moves have similar scores, you can:

   * Prefer moves that:

     * Capture something.
     * Move master to safer squares (king-safety heuristic).
     * Use “more flexible” cards (leave you more options later).
   * Or inject a tiny bit of randomness:

     * Avoid a perfectly deterministic agent if you want diversity in self-play.

4. **Move filtering**

   * Optionally ignore “obviously terrible” moves:

     * E.g., moves that expose your master to immediate capture by a simple static check.
   * This can speed up 1-ply and improve move quality.

---

### 3. Experiments & knobs at the 1-ply level

1. **Greedy vs “softmax” move selection**

   * Greedy: always pick best (V(s_a)).
   * Softmax:
     [
     P(a|s) \propto \exp(\beta V(s_a))
     ]

     * Temperature (\beta) controls exploration.
     * This is a nice bridge to RL: same form as a Boltzmann policy on a value estimate.

2. **Heuristic shaping**

   * Compare:

     * 0-ply: just choose move that gives best immediate heuristic change without forward simulation (if you have such variant).
     * 1-ply lookahead: simulate and measure (V(s_a)).
   * You’ll see how much “just one lookahead” helps.

3. **Computational budget**

   * Onitama’s branching factor is small (often < 20 moves), so full 1-ply is cheap.
   * But you can also think about a time control:

     * “Agent must move within X ms; if too many moves, consider pruning.”

4. **Sensitivity to heuristic weights**

   * Use 1-ply agent as your **microscope** for heuristic design:

     * Small changes in weights can change preferred move a lot.
     * This tells you which features matter in actual decision-making.

**Outcome of Step C:**
You have a “search-enhanced heuristic player” and a clear view of the value of one-step lookahead.

---

## D. Shallow Minimax / Alpha-Beta – adversarial search

### 1. Intuition

1-ply assumes the opponent doesn’t react. Minimax assumes they **do their best to hurt you**.

At depth-2 (your move, their move) the idea is:

* You choose a move (a).
* Assume opponent replies with the move that **minimizes** your resulting value.
* You pick the move whose worst-case outcome is best.

Formally (depth-2):

[
a^* = \arg\max_{a \in \mathcal{A}(s)} \ \min_{b \in \mathcal{A}(s_a)} V(s_{a,b})
]

where:

* (s_a): state after your move (a)
* (s_{a,b}): state after your move (a) and opponent move (b)

You can extend this to deeper depths with alternating max/min and using (V) only at the leaves.

---

### 2. Minimax with depth limit

In general, for depth (d):

* If (d = 0) or terminal state:

  * Return (V(s)).
* If it’s your turn (maximizing):
  [
  \text{value}(s, d) = \max_{a \in \mathcal{A}(s)} \text{value}(s_a, d-1)
  ]
* If it’s opponent’s turn (minimizing):
  [
  \text{value}(s, d) = \min_{a \in \mathcal{A}(s)} \text{value}(s_a, d-1)
  ]

The agent plays:

[
a^* = \arg\max_{a \in \mathcal{A}(s)} \text{value}(s_a, d-1)
]

This is where your heuristic becomes a **leaf evaluator**.

---

### 3. Design decisions for minimax

1. **Depth**

   * Depth=1 → just 1-ply (max).
   * Depth=2 → you move, opponent replies.
   * Depth=3–4 → starts to feel more “chessy,” but compute grows as ~(b^d).
   * Onitama’s small branching factor makes depth 3–4 plausible for offline analysis, maybe not for tight real-time constraints.

2. **Alpha-beta pruning**

   * Classic optimization that doesn’t change the result, just prunes branches that can’t affect the final decision.
   * Important if you go beyond depth=2.

3. **Move ordering**

   * Evaluation order of moves strongly affects alpha-beta’s pruning efficiency.
   * Simple heuristics:

     * Try captures first.
     * Try moves that move master towards central or safer squares.
   * You can reuse 1-ply heuristic scores to order moves.

4. **Transposition table**

   * Onitama has repeated positions (symmetries, card swaps, etc.).
   * Cache evaluated states so you don’t recompute them at deeper depths.
   * This introduces the notion of a **state hashing function** (e.g., Zobrist-like).

5. **Quiescence search (optional)**

   * Idea: avoid evaluating “noisy” positions (e.g., in the middle of big capture sequences) at the depth frontier.
   * If a leaf position has capture moves available, extend search deeper a bit.
   * In Onitama, you might:

     * Extend if the master is in immediate danger.
     * Extend if there are multiple capture moves possible.

6. **Evaluation perspective**

   * Again, simplest: have (V) always return the score from the **player to move** viewpoint.
   * Then minimax doesn’t need sign-flips; max nodes always “current player.”

---

### 4. Experiments to run with minimax

1. **Depth vs strength curve**

   * Run depth-1, 2, 3 agents in round-robin matches:

     * How much does each extra ply help?
     * Is depth=2 already crushing heuristic & 1-ply?
   * You can establish a **difficulty ladder** for future RL agents.

2. **Heuristic quality vs depth**

   * For a weak heuristic, deeper search might overfit noise.
   * For a better heuristic, depth helps more.
   * Useful observation: RL will improve the heuristic, which will make minimax much stronger.

3. **Time / node budget vs Elo**

   * Impose a limit: “Agent may search at most N nodes or T milliseconds per move.”
   * Compare:

     * Depth-fixed vs time-fixed strategies.
   * This is helpful if you ever want to deploy this as “play against AI level 1–5.”

4. **Effect of alpha-beta and move ordering**

   * Implement plain minimax vs alpha-beta + move ordering:

     * Compare nodes searched.
     * Compare actual playing strength for same time budget.

**Outcome of Step D:**
You have a controllable, search-based family of agents (by depth, by node/time budget, by heuristic weights) that define a **baseline “skill ladder”** for the rest of your RL exploration.

---

## How all of this feeds RL later

As you think about design decisions for B/C/D, it’s helpful to view them as scaffolding for RL:

* The heuristic (V(s; w)) is a **natural value function parameterization** for RL.
* 1-ply and minimax agents can:

  * Generate training data (state, action, outcome) for IL / offline RL.
  * Serve as strong opponents for self-play.
* The knobs you built (weights, depth, exploration temperature, etc.) will be:

  * Parameters you compare RL agents against.
  * Useful baselines (“Does PPO at iteration 100 beat depth-2 minimax?”).

---

If you’re up for it next, we can:

* Enumerate a **concrete feature list** for (V(s)) that’s realistic to implement in your existing codebase.
* Sketch an **experiment matrix** like:
  rows = agents (heuristic variants, 1-ply, depth-2, depth-3)
  columns = metrics (win rates vs each other, average length, etc.)
  so you have a clear “Phase 1 experimental plan” even before touching RL.

