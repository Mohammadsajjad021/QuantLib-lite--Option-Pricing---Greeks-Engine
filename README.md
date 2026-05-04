# Quant Option Pricing Engine

![CI](https://github.com/Mohammadsajjad021/QuantLib-lite--Option-Pricing---Greeks-Engine/actions/workflows/ci.yml/badge.svg)

Python library for pricing options and computing Greeks using analytical and numerical methods.

---

## Features

* Black–Scholes pricing (call & put)
* Monte Carlo simulation
* Binomial pricing 
* Greeks: Delta, Gamma, Vega, Theta, Rho, etc.
* Implied volatility (Newton–Raphson)
* Market calibration from option quotes
* Unit tests (arbitrage checks, finite differences)
* CI with GitHub Actions

---

## Installation

```bash
git clone https://github.com/Mohammadsajjad021/QuantLib-lite--Option-Pricing---Greeks-Engine.git
cd YOUR_REPO
pip install -r requirements.txt
```

---

## Usage

```python
from quant_pricing.black_scholes import call_price
from quant_pricing.implied_vol import implied_vol

price = call_price(100, 100, 1, 0.05, 0.2)
iv = implied_vol(price, 100, 100, 1, 0.05)

dividend_price = call_price(100, 100, 1, 0.05, 0.2, q=0.02)
```

### Market calibration

```python
from calibration import calibrate_flat_volatility, calibrate_implied_vols

quotes = [
    {"K": 90, "T": 0.5, "market_price": 13.50},
    {"K": 100, "T": 1.0, "market_price": 10.45},
    {"K": 110, "T": 1.5, "market_price": 12.25, "type": "put"},
]

surface = calibrate_implied_vols(quotes, S=100, r=0.05, q=0.02)
flat_vol = calibrate_flat_volatility(quotes, S=100, r=0.05, q=0.02)
```

---

## Run Tests

```bash
python3 -m pytest
```

---

## Project Structure

```text
src/quant_pricing/   # pricing models
tests/               # unit tests
notebooks/           # demos & visualizations
```

---

## Highlights

* Monte Carlo vs Black–Scholes validation
* Greeks verified via finite differences
* Implied volatility recovery
* Volatility smile (notebook)
