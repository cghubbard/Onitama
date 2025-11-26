#!/usr/bin/env python3
"""
Entry point for running the Onitama web server.

Usage:
    python run_web.py [--host HOST] [--port PORT] [--reload]

Examples:
    python run_web.py                    # Run on localhost:8000
    python run_web.py --port 3000        # Run on localhost:3000
    python run_web.py --host 0.0.0.0     # Allow external connections
    python run_web.py --reload           # Auto-reload on code changes
"""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run the Onitama web server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes"
    )

    args = parser.parse_args()

    print(f"Starting Onitama web server at http://{args.host}:{args.port}")
    print(f"API documentation: http://{args.host}:{args.port}/docs")
    print()

    uvicorn.run(
        "src.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
