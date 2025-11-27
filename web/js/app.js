/**
 * Main application for Onitama web interface.
 */

// Global game controller
let gameController = null;

// DOM Elements
const elements = {
    setupPanel: null,
    gameArea: null,
    replayPanel: null,
    gameOverOverlay: null,
    blueAgentSelect: null,
    redAgentSelect: null,
    startGameBtn: null,
    newGameBtn: null,
    aiControls: null
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    // Cache DOM elements
    elements.setupPanel = document.getElementById('setup-panel');
    elements.gameArea = document.getElementById('game-area');
    elements.replayPanel = document.getElementById('replay-panel');
    elements.gameOverOverlay = document.getElementById('game-over-overlay');
    elements.blueAgentSelect = document.getElementById('blue-agent');
    elements.redAgentSelect = document.getElementById('red-agent');
    elements.startGameBtn = document.getElementById('start-game');
    elements.newGameBtn = document.getElementById('new-game-btn');
    elements.aiControls = document.getElementById('ai-controls');

    // Initialize game controller
    gameController = new GameController();
    await gameController.init();

    // Set up event listeners
    setupEventListeners();

    // Show setup panel
    showSetup();
});

function setupEventListeners() {
    // Start game button
    if (elements.startGameBtn) {
        elements.startGameBtn.addEventListener('click', startGame);
    }

    // New game button
    if (elements.newGameBtn) {
        elements.newGameBtn.addEventListener('click', () => {
            gameController.disconnect();
            gameController.hideGameOver();
            showSetup();
        });
    }

    // AI controls
    const startAIBtn = document.getElementById('start-ai-btn');
    const pauseBtn = document.getElementById('pause-btn');
    const resumeBtn = document.getElementById('resume-btn');
    const stepBtn = document.getElementById('step-btn');
    const speedSlider = document.getElementById('speed-slider');

    if (startAIBtn) {
        startAIBtn.addEventListener('click', () => {
            gameController.startAIGame();
            startAIBtn.disabled = true;
        });
    }

    if (pauseBtn) {
        pauseBtn.addEventListener('click', () => gameController.pauseGame());
    }

    if (resumeBtn) {
        resumeBtn.addEventListener('click', () => gameController.resumeGame());
    }

    if (stepBtn) {
        stepBtn.addEventListener('click', () => gameController.stepGame());
    }

    if (speedSlider) {
        speedSlider.addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            gameController.setAISpeed(speed);
            const label = document.getElementById('speed-label');
            if (label) label.textContent = `${speed.toFixed(1)}s`;
        });
    }

    // Tab switching
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.target;
            switchTab(target);
        });
    });
}

async function startGame() {
    const blueAgent = elements.blueAgentSelect.value;
    const redAgent = elements.redAgentSelect.value;

    elements.startGameBtn.disabled = true;
    elements.startGameBtn.textContent = 'Starting...';

    try {
        await gameController.createGame(blueAgent, redAgent);
        showGame();

        // Show AI controls if both players are AI
        if (blueAgent !== 'human' && redAgent !== 'human') {
            showAIControls();
        } else {
            hideAIControls();
        }

        // If it's AI's turn first, request a move
        if (!gameController.state.isHumanTurn()) {
            setTimeout(() => {
                gameController.requestAIMove();
            }, 500);
        }
    } catch (e) {
        alert('Failed to start game: ' + e.message);
    } finally {
        elements.startGameBtn.disabled = false;
        elements.startGameBtn.textContent = 'Start Game';
    }
}

function showSetup() {
    elements.setupPanel.classList.remove('hidden');
    elements.gameArea.classList.add('hidden');
    if (elements.replayPanel) elements.replayPanel.classList.add('hidden');

    // Hide replay controls when showing setup
    const replayControls = document.getElementById('replay-controls');
    if (replayControls) replayControls.classList.add('hidden');

    // Clear replay info
    const replayInfo = document.getElementById('replay-info');
    if (replayInfo) replayInfo.textContent = '';
}

function showGame() {
    elements.setupPanel.classList.add('hidden');
    elements.gameArea.classList.remove('hidden');
    if (elements.replayPanel) elements.replayPanel.classList.add('hidden');
    gameController.render();
}

function showAIControls() {
    if (elements.aiControls) {
        elements.aiControls.classList.remove('hidden');
        // Reset button states
        const startAIBtn = document.getElementById('start-ai-btn');
        if (startAIBtn) startAIBtn.disabled = false;
    }
}

function hideAIControls() {
    if (elements.aiControls) {
        elements.aiControls.classList.add('hidden');
    }
}

function switchTab(target) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.target === target);
    });

    // Show/hide panels
    if (target === 'play') {
        showSetup();
    } else if (target === 'replays') {
        loadReplays();
    }
}

async function loadReplays() {
    elements.setupPanel.classList.add('hidden');
    elements.gameArea.classList.add('hidden');
    if (elements.replayPanel) {
        elements.replayPanel.classList.remove('hidden');
    }

    try {
        const response = await fetch('/api/replays?limit=50');
        const data = await response.json();
        renderReplayList(data.games);
    } catch (e) {
        console.error('Failed to load replays:', e);
    }
}

function renderReplayList(games) {
    const listEl = document.getElementById('replay-list');
    if (!listEl) return;

    listEl.innerHTML = '';

    if (games.length === 0) {
        listEl.innerHTML = '<p style="text-align: center; color: #888;">No replays available</p>';
        return;
    }

    games.forEach(game => {
        const item = document.createElement('div');
        item.className = 'replay-item';
        item.dataset.gameId = game.game_id;

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

        item.addEventListener('click', () => loadReplay(game.game_id));

        listEl.appendChild(item);
    });
}

async function loadReplay(gameId) {
    try {
        const response = await fetch(`/api/replays/${gameId}`);
        const trajectory = await response.json();

        // Initialize replay viewer
        initReplayViewer(trajectory);
    } catch (e) {
        console.error('Failed to load replay:', e);
        alert('Failed to load replay');
    }
}

let replayData = null;
let replayIndex = 0;
let replayListenersInitialized = false;

function initReplayViewer(trajectory) {
    replayData = trajectory;
    replayIndex = 0;

    // Show game area for replay
    elements.setupPanel.classList.add('hidden');
    elements.gameArea.classList.remove('hidden');
    hideAIControls();

    // Show replay controls
    const replayControls = document.getElementById('replay-controls');
    if (replayControls) replayControls.classList.remove('hidden');

    // Set up replay slider
    const slider = document.getElementById('replay-slider');
    if (slider) {
        slider.max = trajectory.transitions.length - 1;
        slider.value = 0;
    }

    // Only set up event listeners once to prevent duplicate handlers
    if (!replayListenersInitialized) {
        replayListenersInitialized = true;

        if (slider) {
            slider.addEventListener('input', (e) => {
                replayIndex = parseInt(e.target.value);
                renderReplayState();
            });
        }

        // Set up replay buttons
        document.getElementById('replay-start')?.addEventListener('click', () => {
            replayIndex = 0;
            renderReplayState();
        });

        document.getElementById('replay-prev')?.addEventListener('click', () => {
            if (replayIndex > 0) {
                replayIndex--;
                renderReplayState();
            }
        });

        document.getElementById('replay-next')?.addEventListener('click', () => {
            if (replayData && replayIndex < replayData.transitions.length - 1) {
                replayIndex++;
                renderReplayState();
            }
        });

        document.getElementById('replay-end')?.addEventListener('click', () => {
            if (replayData) {
                replayIndex = replayData.transitions.length - 1;
                renderReplayState();
            }
        });
    }

    renderReplayState();
}

function renderReplayState() {
    if (!replayData || !replayData.transitions[replayIndex]) return;

    const transition = replayData.transitions[replayIndex];
    const state = transition.state;

    // Update game controller state
    gameController.state.update({
        board: state.board,
        current_player: state.current_player,
        blue_cards: state.blue_cards,
        red_cards: state.red_cards,
        neutral_card: state.neutral_card,
        outcome: state.outcome,
        move_count: state.move_number,
        blue_agent: replayData.config.blue_agent,
        red_agent: replayData.config.red_agent
    });

    // Disable player interaction for replay
    gameController.boardRenderer.setPlayerTurn(false, null);

    // Render the board
    gameController.render();

    // Highlight the move that was made
    if (transition.action) {
        gameController.boardRenderer.highlightReplayMove(
            transition.action.from,
            transition.action.to
        );
    }

    // Update slider
    const slider = document.getElementById('replay-slider');
    if (slider) slider.value = replayIndex;

    // Update info
    const info = document.getElementById('replay-info');
    if (info) {
        info.textContent = `Move ${replayIndex + 1} of ${replayData.transitions.length}`;
        if (transition.action) {
            const action = transition.action;
            info.textContent += ` - ${action.card}: (${action.from.join(',')}) → (${action.to.join(',')})`;
        }
    }
}
