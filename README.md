# Crude Oil Price Risk: Bayesian Networks & Regime-Switching Models

Mini-capstone project for **MScFE 660 Risk Management** (WorldQuant
University), completed across three Group Work Projects (GWP1-3). The
project replicates, critiques, and extends the Bayesian-network approach to
forecasting the price of crude oil described in:

> Alvi, Danish A. *Application of Probabilistic Graphical Models in
> Forecasting Crude Oil Price*. 2018. University College London,
> Dissertation. <https://arxiv.org/abs/1804.10869>

## Project structure

```
oil-price-risk-capstone/
├── data/                   # raw / interim / processed / external datasets
├── docs/
│   └── AGENT_BRIEF.md      # full task specification and execution plan
├── notebooks/
│   ├── GWP1_problem_formulation_and_data.ipynb
│   ├── GWP2_methodology_and_model_development.ipynb
│   └── GWP3_results_and_interpretation.ipynb
├── reports/
│   ├── GWP1/ GWP2/ GWP3/   # one LaTeX project per GWP (main.tex, references.bib, figures/)
├── src/oilbn/              # installable Python package (OOP source)
│   ├── data/               # loaders, cleaners, data dictionary registry
│   ├── eda/                # distribution / time-series / multivariate plotters
│   ├── models/             # HMM regime model, Bayesian network, Black-Litterman
│   ├── evaluation/         # train/val/test splitting, forecast evaluation
│   └── utils/              # configuration helpers
└── tests/                  # pytest unit tests mirroring src/oilbn
```

## Setup

```powershell
# 1. Create/activate the virtual environment (already present as .venv at the
#    workspace root in this repo's parent folder; adjust path as needed).
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install pinned dependencies, then the package itself (editable).
pip install -r requirements-dev.txt   # or requirements.txt for runtime-only
pip install -e .

# 3. Configure optional API keys (not required: loaders fall back to
#    keyless public endpoints when unset).
copy .env.example .env

# 4. Run the test suite.
pytest

# 5. Launch notebooks.
jupyter lab notebooks/
```

`requirements.txt` pins the runtime dependencies (`numpy`, `pandas`,
`pgmpy`, etc.); `requirements-dev.txt` additionally pins testing/linting/
notebook tooling (`pytest`, `ruff`, `black`, `jupyter`, ...). Both mirror
`pyproject.toml`'s `dependencies` / `[project.optional-dependencies].dev`
lists but with exact, reproducible version pins. Alternatively, install
directly from `pyproject.toml` with `pip install -e ".[dev]"`.

## Building the LaTeX reports

Each `reports/GWP{n}/` folder is a self-contained LaTeX project using
`biblatex` with the MLA citation style. Requires a TeX distribution (MiKTeX
or TeX Live) with `pdflatex` and `biber` on PATH:

```powershell
cd reports/GWP1
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Development conventions

- Strict object-oriented design: every responsibility (data loading,
  cleaning, plotting, modeling, evaluation) is a class with a single
  purpose; see `docs/AGENT_BRIEF.md` for the full architecture.
- Code style enforced with `black` and `ruff` (`pyproject.toml`).
- Unit tests under `tests/` use `pytest`, with network calls mocked.
- One topic branch per GWP (`gwp1-data`, `gwp2-methodology`,
  `gwp3-results`), merged into `main` as each phase completes.

## Authors

Student 1 · Student 2 · Student 3 — MScFE 660 Risk Management Group Work
Project, WorldQuant University.
