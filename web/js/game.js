/**
 * Game state management for Onitama.
 */

class GameState {
    constructor() {
        this.gameId = null;
        this.board = {};
        this.currentPlayer = 0;
        this.blueCards = [];
        this.redCards = [];
        this.neutralCard = '';
        this.outcome = 0;
        this.moveCount = 0;
        this.blueAgent = 'human';
        this.redAgent = 'heuristic';
        this.legalMoves = [];
    }

    update(state) {
        if (state.game_id) this.gameId = state.game_id;
        if (state.board) this.board = state.board;
        if (state.current_player !== undefined) this.currentPlayer = state.current_player;
        if (state.blue_cards) this.blueCards = state.blue_cards;
        if (state.red_cards) this.redCards = state.red_cards;
        if (state.neutral_card) this.neutralCard = state.neutral_card;
        if (state.outcome !== undefined) this.outcome = state.outcome;
        if (state.move_count !== undefined) this.moveCount = state.move_count;
        if (state.blue_agent) this.blueAgent = state.blue_agent;
        if (state.red_agent) this.redAgent = state.red_agent;
    }

    setLegalMoves(moves) {
        this.legalMoves = moves;
    }

    isGameOver() {
        return this.outcome !== 0;
    }

    getWinner() {
        if (this.outcome === 1) return 0; // Blue wins
        if (this.outcome === 2) return 1; // Red wins
        return null; // Draw or ongoing
    }

    isHumanTurn() {
        if (this.currentPlayer === 0) {
            return this.blueAgent === 'human';
        }
        return this.redAgent === 'human';
    }

    getCurrentPlayerAgent() {
        return this.currentPlayer === 0 ? this.blueAgent : this.redAgent;
    }

    getCurrentCards() {
        return this.currentPlayer === 0 ? this.blueCards : this.redCards;
    }
}

// Game controller
class GameController {
    constructor() {
        this.state = new GameState();
        this.ws = null;
        this.boardRenderer = null;
        this.cardRenderer = null;
        this.selectedCard = null;
        this.playerRole = null; // 0 = blue, 1 = red, null = spectator
        this.isPaused = false;
        this.aiSpeed = 1.0;
    }

    async init() {
        this.boardRenderer = new BoardRenderer('board');
        this.cardRenderer = new CardRenderer();

        // Load card data
        await this.loadCards();

        // Set up event handlers
        this.boardRenderer.onMoveCallback = (from, to, card) => {
            this.makeMove(from, to, card);
        };
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

    async createGame(blueAgent, redAgent) {
        try {
            const response = await fetch('/api/games', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    blue_agent: blueAgent,
                    red_agent: redAgent
                })
            });

            const data = await response.json();
            this.state.update(data.state);

            // Determine player role
            if (blueAgent === 'human' && redAgent !== 'human') {
                this.playerRole = 0;
            } else if (redAgent === 'human' && blueAgent !== 'human') {
                this.playerRole = 1;
            } else if (blueAgent === 'human' && redAgent === 'human') {
                this.playerRole = 0; // Blue goes first in local play
            } else {
                this.playerRole = null; // Spectator for AI vs AI
            }

            // Connect WebSocket
            this.connectWebSocket(this.state.gameId);

            return data;
        } catch (e) {
            console.error('Failed to create game:', e);
            throw e;
        }
    }

    connectWebSocket(gameId) {
        this.ws = new GameWebSocket(gameId, {
            onConnect: () => {
                console.log('Connected to game');
                this.ws.getMoves();
            },
            legal_moves: (data) => {
                this.state.setLegalMoves(data.moves);
                this.boardRenderer.setLegalMoves(data.moves);
                this.render();
            },
            move_made: (data) => {
                console.log('Move made:', data);
                // State will be updated by the state message
                // After state updates, check if we need to request AI move
            },
            state: (data) => {
                this.state.update(data.state);
                this.render();

                // Request legal moves if it's human turn
                if (this.state.isHumanTurn() && !this.state.isGameOver()) {
                    this.ws.getMoves();
                }

                // Request AI move if it's AI turn
                if (!this.state.isHumanTurn() && !this.state.isGameOver()) {
                    setTimeout(() => {
                        this.requestAIMove();
                    }, 500);
                }
            },
            game_over: (data) => {
                this.showGameOver(data.winner, data.reason);
            },
            error: (data) => {
                this.showError(data.message);
            },
            game_paused: () => {
                this.isPaused = true;
                this.updateAIControls();
            },
            game_resumed: () => {
                this.isPaused = false;
                this.updateAIControls();
            },
            ai_game_started: (data) => {
                console.log('AI game started at speed:', data.speed);
            },
            pong: () => {
                console.log('Pong received');
            }
        });

        this.ws.connect();
    }

    render() {
        // Update board
        this.boardRenderer.setPlayerTurn(
            this.state.isHumanTurn() && !this.state.isGameOver(),
            this.playerRole
        );
        this.boardRenderer.render(this.state);

        // Update cards
        this.renderCards();

        // Update game info
        this.updateGameInfo();
    }

    renderCards() {
        const isHumanTurn = this.state.isHumanTurn();
        const currentPlayer = this.state.currentPlayer;

        // Blue cards (top)
        this.cardRenderer.renderCards(
            'blue-cards',
            this.state.blueCards,
            0,
            currentPlayer === 0 ? this.selectedCard : null,
            currentPlayer === 0 && isHumanTurn && this.playerRole === 0,
            (card) => this.selectCard(card)
        );

        // Red cards (bottom)
        this.cardRenderer.renderCards(
            'red-cards',
            this.state.redCards,
            1,
            currentPlayer === 1 ? this.selectedCard : null,
            currentPlayer === 1 && isHumanTurn && this.playerRole === 1,
            (card) => this.selectCard(card)
        );

        // Neutral card
        const neutralContainer = document.getElementById('neutral-card');
        neutralContainer.innerHTML = '';
        const neutralCardEl = this.cardRenderer.renderCard(
            this.state.neutralCard,
            -1, // neutral
            false,
            false,
            () => {}
        );
        neutralContainer.appendChild(neutralCardEl);
    }

    selectCard(cardName) {
        this.selectedCard = cardName;
        this.boardRenderer.selectCard(cardName);
        this.renderCards();
        this.boardRenderer.updateSelection(this.state);
    }

    makeMove(from, to, card) {
        if (this.ws) {
            this.ws.makeMove(from, to, card);
            this.selectedCard = null;
            this.boardRenderer.clearSelection();
        }
    }

    requestAIMove() {
        if (this.ws && !this.state.isHumanTurn()) {
            this.ws.requestAIMove();
        }
    }

    startAIGame() {
        if (this.ws && this.playerRole === null) {
            this.ws.startAIGame(this.aiSpeed);
        }
    }

    pauseGame() {
        if (this.ws) {
            this.ws.pause();
        }
    }

    resumeGame() {
        if (this.ws) {
            this.ws.resume();
        }
    }

    stepGame() {
        if (this.ws) {
            this.ws.step();
        }
    }

    setAISpeed(speed) {
        this.aiSpeed = speed;
    }

    updateGameInfo() {
        const currentPlayerEl = document.getElementById('current-player');
        const moveCountEl = document.getElementById('move-count');

        if (currentPlayerEl) {
            currentPlayerEl.textContent = this.state.currentPlayer === 0 ? 'BLUE' : 'RED';
            currentPlayerEl.className = this.state.currentPlayer === 0 ? 'blue' : 'red';
        }

        if (moveCountEl) {
            moveCountEl.textContent = `Move: ${this.state.moveCount}`;
        }
    }

    updateAIControls() {
        const pauseBtn = document.getElementById('pause-btn');
        const resumeBtn = document.getElementById('resume-btn');

        if (pauseBtn && resumeBtn) {
            pauseBtn.disabled = this.isPaused;
            resumeBtn.disabled = !this.isPaused;
        }
    }

    showGameOver(winner, reason) {
        const overlay = document.getElementById('game-over-overlay');
        const panel = document.getElementById('game-over-panel');
        const title = panel.querySelector('h2');
        const reasonEl = panel.querySelector('.reason');

        panel.className = '';
        if (winner === 0) {
            panel.classList.add('blue-wins');
            title.textContent = 'BLUE WINS!';
        } else if (winner === 1) {
            panel.classList.add('red-wins');
            title.textContent = 'RED WINS!';
        } else {
            title.textContent = 'DRAW';
        }

        if (reasonEl) {
            const reasonText = {
                'master_captured': 'Master was captured',
                'shrine_reached': 'Master reached the shrine',
                'draw': 'Game ended in a draw',
                'max_moves': 'Maximum moves reached'
            };
            reasonEl.textContent = reasonText[reason] || reason;
        }

        overlay.classList.remove('hidden');
    }

    hideGameOver() {
        const overlay = document.getElementById('game-over-overlay');
        overlay.classList.add('hidden');
    }

    showError(message) {
        const statusEl = document.getElementById('status-message');
        if (statusEl) {
            statusEl.textContent = message;
            statusEl.className = 'error';
            statusEl.classList.remove('hidden');

            setTimeout(() => {
                statusEl.classList.add('hidden');
            }, 3000);
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.disconnect();
            this.ws = null;
        }
    }
}

// Export
window.GameState = GameState;
window.GameController = GameController;
