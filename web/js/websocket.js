/**
 * WebSocket client for Onitama game communication.
 */

class GameWebSocket {
    constructor(gameId, handlers = {}) {
        this.gameId = gameId;
        this.handlers = handlers;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.gameId}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            if (this.handlers.onConnect) {
                this.handlers.onConnect();
            }
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            if (this.handlers.onDisconnect) {
                this.handlers.onDisconnect(event);
            }
            this.attemptReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            if (this.handlers.onError) {
                this.handlers.onError(error);
            }
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (e) {
                console.error('Failed to parse message:', e);
            }
        };
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Reconnecting (attempt ${this.reconnectAttempts})...`);
            setTimeout(() => this.connect(), this.reconnectDelay * this.reconnectAttempts);
        }
    }

    handleMessage(data) {
        const type = data.type;
        const handler = this.handlers[type];

        if (handler) {
            handler(data);
        } else {
            console.log('Unhandled message type:', type, data);
        }
    }

    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.error('WebSocket not connected');
        }
    }

    // Game actions
    getState() {
        this.send({ type: 'get_state' });
    }

    getMoves() {
        this.send({ type: 'get_moves' });
    }

    makeMove(fromPos, toPos, card) {
        this.send({
            type: 'move',
            from_pos: fromPos,
            to_pos: toPos,
            card: card
        });
    }

    requestAIMove() {
        this.send({ type: 'ai_move' });
    }

    startAIGame(speed = 1.0) {
        this.send({ type: 'start_ai_game', speed: speed });
    }

    pause() {
        this.send({ type: 'pause' });
    }

    resume() {
        this.send({ type: 'resume' });
    }

    step() {
        this.send({ type: 'step' });
    }

    ping() {
        this.send({ type: 'ping' });
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

// Export for use in other modules
window.GameWebSocket = GameWebSocket;
