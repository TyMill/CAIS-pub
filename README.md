# MARIS-AI (v0.2.1) — CAIS Experimental Stack + Statistics/Report

End-to-end, self-contained experimental stack for:
- multi-agent maritime-inspired encounters (2D),
- governance-constrained decision execution (**G**),
- audit trace semantics (**Φ**) with hash chaining,
- replayability verification (**Ψ**),
- federated learning (FedAvg),
- Results-ready exports (CSV/JSON/PNG) + statistical comparisons + LaTeX tables.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Sweep governance modes
```bash
maris-cais-sweep --episodes 30 --steps 200 --agents 5 --seed 42 --scenario crossing --centralized --projection slsqp
```

## Plots
```bash
maris-cais-plots outputs/
```

## Integrity replay (Ψ)
```bash
maris-cais-replay outputs/<run_id>/
```

## Federated run
```bash
maris-cais-federated --rounds 20 --clients 10 --agents 5 --seed 7 --mode project --projection slsqp
```

## Paper-ready report (stats + LaTeX)
```bash
maris-cais-report outputs/ --metric violation_rate
```
Outputs:
- `outputs/reports/summary_with_ci.csv`
- `outputs/reports/pairwise_tests.csv`
- `outputs/reports/table_summary.tex`
- `outputs/reports/table_pairwise.tex`
