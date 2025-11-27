# Linear Heuristic Value Function – Training Plan

This document describes the procedure used to learn a linear value function for evaluating Onitama board states. The value function is trained from recorded game trajectories using weighted logistic regression with optional L1/L2 regularization.

---

## 1. Overview

We train a linear evaluation function of the form:

\[
V(s) = \theta^\top \tilde{\phi}(s) + b
\]

where:

- \( \phi(s) \) is a hand-crafted feature vector extracted from state \(s\)
- \( \tilde{\phi}(s) \) is a standardized version of \( \phi(s) \)
- \( V(s) \) represents the **log-odds** that the current player eventually wins the game

This value function is used by heuristic, 1-ply, and minimax agents to score positions.

---

## 2. Training Data

Training examples are generated from recorded self-play games.

For each game \(g\) with states:

\[
s^{(g)}_0, s^{(g)}_1, \dots, s^{(g)}_{H_g}
\]

we use **all non-terminal states** \( s^{(g)}_0 \dots s^{(g)}_{H_g - 1} \).  
Terminal states are excluded from training.

For each state we extract:

- Feature vector \( \phi^{(g)}_t \)
- Outcome label \( o^{(g)}_t \in \{+1, -1\} \), from the **current player’s perspective**:
  - \(+1\): current player eventually wins  
  - \(-1\): current player eventually loses
- A time-based weight \( w^{(g)}_t \) (defined below)

---

## 3. Time-Based Sample Weights

Later states should influence training more strongly than early states.

For a game of length \(H_g\), define:

\[
w^{(g)}_t = \gamma^{H_g - 1 - t}
\]

with discount factor:

- **\(\gamma = 0.97\)** (initial default)

Thus:

- The last non-terminal state receives weight \(1.0\)
- Early states receive progressively smaller weights

---

## 4. Feature Standardization

Each raw feature dimension is standardized using training-set statistics:

\[
\tilde{\phi}_{k} = \frac{\phi_{k} - \mu_k}{\sigma_k + \epsilon}
\]

where:

- \(\mu_k\): mean of feature \(k\) across training samples  
- \(\sigma_k\): standard deviation of feature \(k\) across training samples  
- \(\epsilon\): small constant to avoid division by zero

The same transform is applied at inference time.

---

## 5. Logistic Value Model

For each training sample:

- **Logit (state value):**

  \[
  z = \theta^\top \tilde{\phi} + b
  \]

- **Predicted win probability:**

  \[
  p = \sigma(z) = \frac{1}{1 + e^{-z}}
  \]

- **Target label encoding:**
  - \(y = 1\) if \(o = +1\)  
  - \(y = 0\) if \(o = -1\)

---

## 6. Training Objective

We minimize the **weighted logistic loss** with optional L1 and L2 regularization:

\[
\mathcal{L}(\theta, b)
=
\sum_{(g,t)}
w^{(g)}_t 
\,
\ell\!\left(z^{(g)}_t, y^{(g)}_t\right)
+
\lambda_2 \|\theta\|_2^2
+
\lambda_1 \|\theta\|_1
\]

where:

- \( \ell \) is binary cross-entropy  
- \( \lambda_2 \) controls L2 regularization  
- \( \lambda_1 \) controls L1 regularization  

Different settings of \( (\lambda_1, \lambda_2) \) produce different heuristic agents.

---

## 7. Train/Validation Split

Cross-validation is performed at the **game level** (not per state):

1. Shuffle recorded games  
2. Divide into training and validation folds  
3. Train on complete trajectories of games in the training set  
4. Evaluate weighted log-loss on validation games  
5. Select optimal regularization parameters \( (\lambda_1, \lambda_2) \)

---

## 8. Output

Training produces:

- Weight vector \( \theta \in \mathbb{R}^d \)
- Bias term \( b \)
- Feature normalization statistics \( \mu_k, \sigma_k \)

These define the final value function:

\[
V(s) = \theta^\top \tilde{\phi}(s) + b
\]

This function is used by all heuristic, 1-ply, and minimax agents.

---

