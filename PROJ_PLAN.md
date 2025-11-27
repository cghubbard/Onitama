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

## How I’d actually do this, in order

If you want a concrete implementation order:

1. **Improve heuristic + build tournament + logging** (project 1–2 evenings).
2. **Tabular Q-learning on a tiny Onitama variant** to get the Bellman stuff into your fingers.
3. **Linear SARSA/Q with hand-crafted features** on the real game.
4. **REINFORCE + baseline** with a simple CNN policy.
5. **Actor-critic + PPO**, filling in `PPOAgent` and `train_ppo.py`.
6. **MCTS agent** for comparison and as a possible “teacher”.
7. **Imitation + curriculum + self-play league** once PPO is stable.

If you’d like, next step we can pick **one** of these (e.g., tabular Q on a simplified environment) and sketch out the exact state encoding, data structures, and training loop in your current repo layout.

[1]: https://github.com/cghubbard/Onitama "GitHub - cghubbard/Onitama: RL Understand Work"
