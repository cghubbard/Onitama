/**
 * Linear Heuristic Sandbox for Onitama.
 * Enables interactive evaluation of positions from replayed games.
 */

class SandboxController {
    constructor() {
        this.trajectory = null;
        this.moveIndex = 0;
        this.evaluation = null;
        this.selectedMoves = [];  // Up to 2 moves for comparison
        this.boardRenderer = null;
        this.cardRenderer = null;
        this.featureNames = [];
        this.defaultWeights = [];
    }

    async init() {
        // Initialize renderers for sandbox board
        this.boardRenderer = new BoardRenderer('sandbox-board');
        this.cardRenderer = new CardRenderer();

        // Load card movements
        await this.loadCards();

        // Load model info
        await this.loadModelInfo();

        // Set up event listeners
        this.setupEventListeners();
    }

    async loadCards() {
        try {
            const response = await fetch('/api/cards');
            const data = await response.json();
            this.cardRenderer.setCardMovements(data.cards);
        } catch (e) {
            console.error('Failed to load cards:', e);
        }
    }

    async loadModelInfo() {
        try {
            const response = await fetch('/api/sandbox/models');
            const data = await response.json();
            const linearModel = data.models.find(m => m.id === 'linear_heuristic');
            if (linearModel) {
                this.defaultWeights = linearModel.default_weights;
                this.featureNames = linearModel.feature_names;
            }
        } catch (e) {
            console.error('Failed to load model info:', e);
        }
    }

    setupEventListeners() {
        // Navigation
        document.getElementById('sandbox-back')?.addEventListener('click', () => this.showGameList());
        document.getElementById('sandbox-prev')?.addEventListener('click', () => this.prevMove());
        document.getElementById('sandbox-next')?.addEventListener('click', () => this.nextMove());

        // Comparison
        document.getElementById('sandbox-clear-comparison')?.addEventListener('click', () => this.clearComparison());
    }

    async loadGameList() {
        try {
            const response = await fetch('/api/replays?limit=50');
            const data = await response.json();
            this.renderGameList(data.games);
        } catch (e) {
            console.error('Failed to load replays:', e);
        }
    }

    renderGameList(games) {
        const list = document.getElementById('sandbox-replay-list');
        if (!list) return;

        list.innerHTML = '';

        if (games.length === 0) {
            list.innerHTML = '<p class="no-games">No replays available</p>';
            return;
        }

        games.forEach(game => {
            const item = document.createElement('div');
            item.className = 'replay-item';

            const agents = document.createElement('span');
            agents.className = 'agents';
            agents.textContent = `${game.blue_agent} vs ${game.red_agent}`;

            const info = document.createElement('span');
            info.className = 'info';
            info.textContent = `${game.total_moves} moves`;

            const result = document.createElement('span');
            result.className = 'result';
            if (game.winner === 0) {
                result.classList.add('blue');
                result.textContent = 'Blue';
            } else if (game.winner === 1) {
                result.classList.add('red');
                result.textContent = 'Red';
            } else {
                result.classList.add('draw');
                result.textContent = 'Draw';
            }

            item.appendChild(agents);
            item.appendChild(info);
            item.appendChild(result);

            item.addEventListener('click', () => this.loadGame(game.game_id));
            list.appendChild(item);
        });
    }

    async loadGame(gameId) {
        try {
            const response = await fetch(`/api/replays/${gameId}`);
            this.trajectory = await response.json();
            this.moveIndex = 0;
            this.selectedMoves = [];

            // Show evaluation view
            document.getElementById('sandbox-game-list').classList.add('hidden');
            document.getElementById('sandbox-evaluation').classList.remove('hidden');

            await this.evaluatePosition();
        } catch (e) {
            console.error('Failed to load game:', e);
            alert('Failed to load game');
        }
    }

    showGameList() {
        document.getElementById('sandbox-evaluation').classList.add('hidden');
        document.getElementById('sandbox-game-list').classList.remove('hidden');
        document.getElementById('sandbox-comparison').classList.add('hidden');
        this.selectedMoves = [];
    }

    async evaluatePosition() {
        if (!this.trajectory) return;

        try {
            const response = await fetch('/api/sandbox/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    game_id: this.trajectory.game_id,
                    move_number: this.moveIndex,
                    weights: null  // Use default weights
                })
            });

            this.evaluation = await response.json();
            this.renderEvaluation();
        } catch (e) {
            console.error('Failed to evaluate position:', e);
        }
    }

    renderEvaluation() {
        if (!this.evaluation || !this.trajectory) return;

        // Update move info
        const moveInfo = document.getElementById('sandbox-move-info');
        if (moveInfo) {
            moveInfo.textContent = `Move ${this.moveIndex + 1} of ${this.trajectory.transitions.length}`;
        }

        // Render board state
        const transition = this.trajectory.transitions[this.moveIndex];
        this.renderBoard(transition.state);

        // Render current position score
        const scoreEl = document.getElementById('sandbox-current-score');
        if (scoreEl) {
            const score = this.evaluation.current_position_score;
            scoreEl.textContent = score.toFixed(2);
            scoreEl.className = 'score-display';
            if (score > 0) scoreEl.classList.add('positive');
            else if (score < 0) scoreEl.classList.add('negative');
            else scoreEl.classList.add('neutral');
        }

        // Render feature breakdown
        this.renderFeatures('sandbox-current-features', this.evaluation.current_position_features);

        // Render move evaluations
        this.renderMoves();

        // Update comparison if moves are selected
        if (this.selectedMoves.length === 2) {
            this.renderComparison();
        }
    }

    renderBoard(state) {
        // Create a state object for the board renderer
        const boardState = {
            board: state.board,
            current_player: state.current_player
        };

        // Disable player interaction for sandbox
        this.boardRenderer.setPlayerTurn(false, null);
        this.boardRenderer.render(boardState);

        // Render cards
        this.cardRenderer.renderCards('sandbox-blue-cards', state.blue_cards, 0, null, false, () => {});
        this.cardRenderer.renderCards('sandbox-red-cards', state.red_cards, 1, null, false, () => {});

        // Render neutral card
        const neutralContainer = document.getElementById('sandbox-neutral-card');
        if (neutralContainer) {
            neutralContainer.innerHTML = '';
            const neutralCard = this.cardRenderer.renderCard(state.neutral_card, -1, false, false, () => {});
            neutralContainer.appendChild(neutralCard);
        }
    }

    renderFeatures(containerId, features) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = '';

        features.forEach(f => {
            const row = document.createElement('div');
            row.className = 'feature-row';

            const contribution = f.contribution;
            const maxContrib = 100;  // For bar scaling
            const barWidth = Math.min(Math.abs(contribution) / maxContrib * 100, 100);

            row.innerHTML = `
                <span class="feature-name">${this.formatFeatureName(f.feature_name)}</span>
                <div class="feature-bar-container">
                    <div class="feature-bar ${contribution >= 0 ? 'positive' : 'negative'}"
                         style="width: ${barWidth}%"></div>
                </div>
                <span class="feature-value">${contribution.toFixed(1)}</span>
                <span class="feature-detail">(${f.feature_value.toFixed(1)} x ${f.weight.toFixed(1)})</span>
            `;
            container.appendChild(row);
        });
    }

    renderMoves() {
        const container = document.getElementById('sandbox-move-scores');
        if (!container || !this.evaluation) return;

        container.innerHTML = '';

        this.evaluation.moves.forEach((move, index) => {
            const item = document.createElement('div');
            item.className = 'move-item';
            item.dataset.index = index;

            if (move.is_winning_move) item.classList.add('winning');
            if (move.is_capture) item.classList.add('capture');

            // Check if selected
            if (this.selectedMoves.includes(index)) {
                item.classList.add('selected');
                item.classList.add(this.selectedMoves.indexOf(index) === 0 ? 'move-a' : 'move-b');
            }

            const score = move.is_winning_move ? 'WIN' : move.total_score.toFixed(2);
            const scoreClass = move.total_score > 0 ? 'positive' : (move.total_score < 0 ? 'negative' : 'neutral');

            item.innerHTML = `
                <span class="move-notation">
                    ${move.card}: (${move.from_pos.join(',')}) &rarr; (${move.to_pos.join(',')})
                    ${move.is_capture ? '<span class="tag capture-tag">capture</span>' : ''}
                    ${move.is_winning_move ? '<span class="tag win-tag">WIN</span>' : ''}
                </span>
                <span class="move-score ${scoreClass}">${score}</span>
            `;

            item.addEventListener('click', () => this.selectMove(index));
            item.addEventListener('mouseenter', () => this.highlightMove(move));
            item.addEventListener('mouseleave', () => this.clearMoveHighlight());

            container.appendChild(item);
        });
    }

    selectMove(index) {
        const idx = this.selectedMoves.indexOf(index);

        if (idx !== -1) {
            // Deselect
            this.selectedMoves.splice(idx, 1);
        } else if (this.selectedMoves.length < 2) {
            // Select
            this.selectedMoves.push(index);
        } else {
            // Replace second selection
            this.selectedMoves[1] = index;
        }

        // Re-render moves to show selection
        this.renderMoves();

        // Show/hide comparison panel
        if (this.selectedMoves.length === 2) {
            this.renderComparison();
            document.getElementById('sandbox-comparison').classList.remove('hidden');
        } else {
            document.getElementById('sandbox-comparison').classList.add('hidden');
        }
    }

    renderComparison() {
        if (this.selectedMoves.length !== 2 || !this.evaluation) return;

        const moveA = this.evaluation.moves[this.selectedMoves[0]];
        const moveB = this.evaluation.moves[this.selectedMoves[1]];

        // Update header
        const header = document.getElementById('sandbox-comparison-header');
        if (header) {
            header.innerHTML = `
                <span class="move-a">A: ${moveA.card} (${moveA.from_pos.join(',')}) &rarr; (${moveA.to_pos.join(',')})</span>
                <span class="vs">vs</span>
                <span class="move-b">B: ${moveB.card} (${moveB.from_pos.join(',')}) &rarr; (${moveB.to_pos.join(',')})</span>
            `;
        }

        // Update table
        const tbody = document.querySelector('#sandbox-comparison-table tbody');
        if (!tbody) return;

        tbody.innerHTML = '';

        // Add score row first
        const scoreRow = document.createElement('tr');
        scoreRow.className = 'score-row';
        const scoreA = moveA.is_winning_move ? Infinity : moveA.total_score;
        const scoreB = moveB.is_winning_move ? Infinity : moveB.total_score;
        const scoreDiff = scoreA - scoreB;
        scoreRow.innerHTML = `
            <td><strong>Total Score</strong></td>
            <td class="${scoreA > 0 ? 'positive' : 'negative'}">${moveA.is_winning_move ? 'WIN' : scoreA.toFixed(2)}</td>
            <td class="${scoreB > 0 ? 'positive' : 'negative'}">${moveB.is_winning_move ? 'WIN' : scoreB.toFixed(2)}</td>
            <td class="${scoreDiff > 0 ? 'positive' : (scoreDiff < 0 ? 'negative' : '')}">${isFinite(scoreDiff) ? scoreDiff.toFixed(2) : '-'}</td>
        `;
        tbody.appendChild(scoreRow);

        // Add feature rows
        for (let i = 0; i < moveA.features.length; i++) {
            const fA = moveA.features[i];
            const fB = moveB.features[i];
            const diff = fA.contribution - fB.contribution;

            const row = document.createElement('tr');
            const diffClass = Math.abs(diff) > 5 ? (diff > 0 ? 'positive' : 'negative') : '';

            row.innerHTML = `
                <td>${this.formatFeatureName(fA.feature_name)}</td>
                <td>${fA.contribution.toFixed(1)}</td>
                <td>${fB.contribution.toFixed(1)}</td>
                <td class="${diffClass}">${diff.toFixed(1)}</td>
            `;
            tbody.appendChild(row);
        }
    }

    clearComparison() {
        this.selectedMoves = [];
        this.renderMoves();
        document.getElementById('sandbox-comparison').classList.add('hidden');
    }

    highlightMove(move) {
        this.boardRenderer.highlightReplayMove(move.from_pos, move.to_pos);
    }

    clearMoveHighlight() {
        // Clear replay highlights
        const cells = document.querySelectorAll('#sandbox-board .cell');
        cells.forEach(cell => {
            cell.classList.remove('replay-from', 'replay-to');
        });
    }

    formatFeatureName(name) {
        return name
            .replace(/_/g, ' ')
            .replace(/diff/gi, '')
            .trim()
            .split(' ')
            .map(w => w.charAt(0).toUpperCase() + w.slice(1))
            .join(' ');
    }

    prevMove() {
        if (this.moveIndex > 0) {
            this.moveIndex--;
            this.selectedMoves = [];
            document.getElementById('sandbox-comparison').classList.add('hidden');
            this.evaluatePosition();
        }
    }

    nextMove() {
        if (this.trajectory && this.moveIndex < this.trajectory.transitions.length - 1) {
            this.moveIndex++;
            this.selectedMoves = [];
            document.getElementById('sandbox-comparison').classList.add('hidden');
            this.evaluatePosition();
        }
    }
}

// Global sandbox controller
window.sandboxController = new SandboxController();
