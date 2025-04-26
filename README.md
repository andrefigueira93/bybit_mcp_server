# Bybit MCP Server

A powerful server that connects Bybit's trading API with Machine Cognition Protocol (MCP) to enable AI-assisted cryptocurrency trading.

![Bybit MCP Server](https://img.shields.io/badge/Bybit-MCP%20Server-blue)
![Python](https://img.shields.io/badge/Python-3.12-brightgreen
)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

Bybit MCP Server is a bridge between Bybit's trading platform and AI systems using the Machine Cognition Protocol (MCP). It provides a set of powerful tools for market analysis, technical indicators calculation, and automated trading operations.

## Features

- **Real-time Market Data**: Access current prices, market depth, and historical data
- **Advanced Technical Analysis**: Calculate indicators like RSI, MACD, ATR, and more
- **Multi-timeframe Analysis**: Analyze market trends across multiple timeframes
- **Automated Trading**: Place, modify, and cancel orders programmatically
- **Risk Management**: Position sizing recommendations based on volatility
- **Support/Resistance Detection**: Identify key price levels for better entry/exit points
- **Docker Support**: Easy deployment with Docker and docker-compose

## Requirements

- Python 3.12+
- Bybit API credentials
- Valid internet connection

## Installation

### Local Installation with Virtualenv

1. Clone the repository:

   ```bash
   git clone https://github.com/andrefigueira93/bybit-mcp-server.git
   cd bybit-mcp-server
   ```

2. Create and activate a virtual environment:

   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   uv pip install -e .
   ```

4. Create a `.env` file with your API credentials:
   ```
   BYBIT_API_KEY=your_api_key
   BYBIT_API_SECRET=your_api_secret
   BYBIT_TESTNET=True  # Set to False for production
   ```

### Using Docker

1. Clone the repository:

   ```bash
   git clone https://github.com/andrefigueira93/bybit-mcp-server.git
   cd bybit-mcp-server
   ```

2. Create a `.env` file with your API credentials (as above)

3. Build and run with Docker Compose:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

## Usage

### Starting the Server

To run the server locally:

```bash
mcp run main.py -t sse
```

The server will start on port 8000 by default. When using Docker, the service is exposed on port 80.

### Available Tools

The server provides the following tools:

| Tool Name                 | Description                                                 |
| ------------------------- | ----------------------------------------------------------- |
| `get_balance`             | Gets the current wallet balance                             |
| `get_ticker`              | Retrieves current price, volume and other data for a symbol |
| `get_klines`              | Gets historical candlestick data for analysis               |
| `analyze_market`          | Performs comprehensive technical analysis on a symbol       |
| `analyze_multi_timeframe` | Performs analysis across multiple timeframes                |
| `place_order`             | Places buy/sell orders with optional take profit/stop loss  |
| `cancel_order`            | Cancels an active order                                     |
| `get_active_orders`       | Lists all active orders                                     |
| `get_active_positions`    | Lists all open positions                                    |

### Example Usage with MCP

When connected to an MCP-compatible client:

```python
# Example of using the MCP client to get a market analysis
analysis = mcp_client.call("analyze_market", symbol="BTCUSDT", intervalo="60")
print(analysis)

# Example of placing a market order
order = mcp_client.call("place_order",
                       symbol="BTCUSDT",
                       side="Buy",
                       order_type="Market",
                       qty=0.001,
                       stop_loss=25000)
print(order)
```

## Project Structure

```
bybit-mcp-server/
├── main.py            # Main server file with MCP tools
├── pyproject.toml     # Project dependencies and metadata
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── .env               # Environment variables (create this yourself)
└── README.md          # This file
```

## Environmental Variables

| Variable           | Description                                      | Default         |
| ------------------ | ------------------------------------------------ | --------------- |
| `BYBIT_API_KEY`    | API key for Bybit                                | None (Required) |
| `BYBIT_API_SECRET` | API secret for Bybit                             | None (Required) |
| `BYBIT_TESTNET`    | Whether to use testnet (True) or mainnet (False) | True            |

## Contributing

Contributions are welcome! If you'd like to improve the Bybit MCP Server, please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. Trading cryptocurrencies involves significant risk and you can lose money. Past performance is not indicative of future results.
