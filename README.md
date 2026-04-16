# Quant Option Pricing Engine

![CI](https://github.com/Mohammadsajjad021/QuantLib-lite--Option-Pricing---Greeks-Engine/actions/workflows/ci.yml/badge.svg)

Python library for pricing European options and computing Greeks using analytical and numerical methods.

---

## Features

* Black–Scholes pricing (call & put)
* Monte Carlo simulation
* Greeks: Delta, Gamma, Vega, Theta, Rho
* Implied volatility (Newton–Raphson)
* Unit tests (arbitrage checks, finite differences)
* CI with GitHub Actions

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
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

---

## Future Work

* Heston model
* Variance reduction
* Volatility surface
