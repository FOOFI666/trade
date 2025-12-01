# Trade tools

Utilities for streaming Binance Futures market data and evaluating signals.

## Installation

1. Create/activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

The WebSocket streaming features require the `websocket-client` package included in the requirements file.

## Usage

- Update `config.py` to set your desired symbols and intervals.
- Run `scanner.py` to stream klines and evaluate signals:

```bash
python scanner.py
```

Ensure your network allows access to the Binance Futures endpoints.
