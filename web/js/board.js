/**
 * Board rendering for Onitama.
 */

class BoardRenderer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.selectedPiece = null;
        this.selectedCard = null;
        this.legalMoves = [];
        this.onMoveCallback = null;
        this.onCardSelectCallback = null;
        this.isPlayerTurn = false;
        this.playerColor = null; // 0 = blue, 1 = red, null = spectator
    }

    render(state) {
        this.container.innerHTML = '';

        // Create 5x5 grid
        for (let y = 0; y < 5; y++) {
            for (let x = 0; x < 5; x++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.dataset.x = x;
                cell.dataset.y = y;

                // Mark shrines
                if (x === 2 && y === 0) {
                    cell.classList.add('shrine-blue');
                } else if (x === 2 && y === 4) {
                    cell.classList.add('shrine-red');
                }

                // Check for piece at this position
                const posKey = `${x},${y}`;
                if (state.board && state.board[posKey]) {
                    const [player, pieceType] = state.board[posKey];
                    const piece = this.createPiece(player, pieceType);
                    cell.appendChild(piece);
                }

                // Add click handler
                cell.addEventListener('click', (e) => this.handleCellClick(x, y, state));

                this.container.appendChild(cell);
            }
        }

        this.highlightLegalMoves(state);
    }

    createPiece(player, pieceType) {
        const piece = document.createElement('div');
        piece.className = 'piece';
        piece.classList.add(player === 0 ? 'blue' : 'red');

        if (pieceType === 1) {
            piece.classList.add('master');
            piece.textContent = player === 0 ? 'B' : 'R';
        } else {
            piece.textContent = player === 0 ? 'b' : 'r';
        }

        return piece;
    }

    handleCellClick(x, y, state) {
        if (!this.isPlayerTurn) return;

        const posKey = `${x},${y}`;
        const piece = state.board[posKey];

        // Check if clicking on own piece
        if (piece && piece[0] === this.playerColor) {
            // Select this piece
            this.selectedPiece = [x, y];
            this.updateSelection(state);
            return;
        }

        // Check if clicking on a legal move destination
        if (this.selectedPiece && this.selectedCard) {
            const isLegal = this.legalMoves.some(move =>
                move.from[0] === this.selectedPiece[0] &&
                move.from[1] === this.selectedPiece[1] &&
                move.to[0] === x &&
                move.to[1] === y &&
                move.card === this.selectedCard
            );

            if (isLegal && this.onMoveCallback) {
                this.onMoveCallback(this.selectedPiece, [x, y], this.selectedCard);
                this.clearSelection();
            }
        }
    }

    selectCard(cardName) {
        this.selectedCard = cardName;
        if (this.onCardSelectCallback) {
            this.onCardSelectCallback(cardName);
        }
    }

    updateSelection(state) {
        // Clear previous highlights
        this.container.querySelectorAll('.cell').forEach(cell => {
            cell.classList.remove('selected', 'legal-move', 'legal-capture');
        });

        // Highlight selected piece
        if (this.selectedPiece) {
            const [sx, sy] = this.selectedPiece;
            const selectedCell = this.container.querySelector(`[data-x="${sx}"][data-y="${sy}"]`);
            if (selectedCell) {
                selectedCell.classList.add('selected');
            }

            // Highlight legal moves for selected piece and card
            if (this.selectedCard) {
                this.legalMoves.forEach(move => {
                    if (move.from[0] === sx && move.from[1] === sy && move.card === this.selectedCard) {
                        const [tx, ty] = move.to;
                        const targetCell = this.container.querySelector(`[data-x="${tx}"][data-y="${ty}"]`);
                        if (targetCell) {
                            const posKey = `${tx},${ty}`;
                            if (state.board[posKey]) {
                                targetCell.classList.add('legal-capture');
                            } else {
                                targetCell.classList.add('legal-move');
                            }
                        }
                    }
                });
            }
        }
    }

    highlightLegalMoves(state) {
        if (this.selectedPiece && this.selectedCard) {
            this.updateSelection(state);
        }
    }

    setLegalMoves(moves) {
        this.legalMoves = moves;
    }

    clearSelection() {
        this.selectedPiece = null;
        this.selectedCard = null;
        this.container.querySelectorAll('.cell').forEach(cell => {
            cell.classList.remove('selected', 'legal-move', 'legal-capture');
        });
    }

    setPlayerTurn(isPlayerTurn, playerColor) {
        this.isPlayerTurn = isPlayerTurn;
        this.playerColor = playerColor;
    }
}

// Card renderer
class CardRenderer {
    constructor() {
        this.cardMovements = {};
    }

    setCardMovements(cards) {
        cards.forEach(card => {
            this.cardMovements[card.name] = card.movements;
        });
    }

    renderCard(cardName, player, isSelected, isClickable, onClick) {
        const card = document.createElement('div');
        card.className = 'card';

        if (player === 0) card.classList.add('blue');
        else if (player === 1) card.classList.add('red');
        else card.classList.add('neutral');

        if (isSelected) card.classList.add('selected');
        if (isClickable) {
            card.classList.add('clickable');
            card.addEventListener('click', () => onClick(cardName));
        }

        // Card name
        const name = document.createElement('div');
        name.className = 'card-name';
        name.textContent = cardName;
        card.appendChild(name);

        // Movement grid
        const grid = document.createElement('div');
        grid.className = 'card-grid';

        const movements = this.cardMovements[cardName] || [];

        for (let y = -2; y <= 2; y++) {
            for (let x = -2; x <= 2; x++) {
                const cell = document.createElement('div');
                cell.className = 'card-cell';

                if (x === 0 && y === 0) {
                    cell.classList.add('center');
                } else {
                    // Check if this is a valid move
                    // For display, we show from the perspective of the player
                    const isMove = movements.some(([dx, dy]) => {
                        // Movements are stored from RED's perspective
                        // For BLUE, they're negated
                        if (player === 0) {
                            return -dx === x && -dy === y;
                        }
                        return dx === x && dy === y;
                    });
                    if (isMove) {
                        cell.classList.add('move');
                    }
                }

                grid.appendChild(cell);
            }
        }

        card.appendChild(grid);
        return card;
    }

    renderCards(containerId, cards, player, selectedCard, isClickable, onClick) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';

        cards.forEach(cardName => {
            const card = this.renderCard(
                cardName,
                player,
                cardName === selectedCard,
                isClickable,
                onClick
            );
            container.appendChild(card);
        });
    }
}

// Export for use in other modules
window.BoardRenderer = BoardRenderer;
window.CardRenderer = CardRenderer;
