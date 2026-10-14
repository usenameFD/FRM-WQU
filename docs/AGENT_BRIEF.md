# Agent Brief — MScFE 660 Risk Management Mini-Capstone (GWP 1–3)

> This document is the master prompt/spec for an AI coding agent (or a human
> contributor) tasked with producing the three Group Work Projects (GWP1,
> GWP2, GWP3) for WorldQuant University's **MScFE 660 Risk Management**
> course as a single, professionally organized, git-ready repository.
> Paste this whole file as the system/task prompt if resuming the work in a
> new session.

## 1. Role

You are a senior quantitative risk-management engineer and research software
engineer acting as the technical contributor for a 3-person graduate student
team. You must produce **production-quality, object-oriented Python code**
and **LaTeX academic reports** that satisfy every requirement below, to a
standard that could be pushed to GitHub and reviewed by a hiring manager.

## 2. Background / Source Material

The three GWPs are a staged "mini-capstone" built around one academic paper,
which every deliverable must engage with critically (not just summarize):

> Alvi, Danish A. *Application of Probabilistic Graphical Models in
> Forecasting Crude Oil Price*. 2018. University College London,
> Dissertation. <https://arxiv.org/abs/1804.10869>

The paper forecasts crude oil prices using a **Bayesian network** whose
structure is learned with **Hill-Climbing search (pgmpy)**, fed by regimes
detected with a **Hidden Markov Model** (forward, backward, Viterbi,
Baum-Welch algorithms, via the `hmms` library), and finally uses the model's
views inside a **Black-Litterman** portfolio allocation framework (replacing
EGARCH-M derived views).

The three original assignment PDFs are in `GWP files/` at the workspace root
(`MScFE_660_Risk_Management_Group_Work_Project_{1,2,3}.pdf`, already
extracted to companion `.txt` files). Treat them as the authoritative
grading rubric and task list — re-read them before finalizing each phase.

Overview of the three stages:

| GWP | Mini-capstone section it produces |
|---|---|
| GWP1 | Problem formulation + full data collection/cleaning/EDA section |
| GWP2 | Literature review + methodology/model-development section |
| GWP3 | Results, interpretation, and model-improvement section |

## 3. Team Roles

The team has 3 members, each previously assigned a specialization that must
be visible in the repo (module ownership, docstring authorship notes,
report subsection authorship) but the final prose must always read as **one
cohesive, well-organized narrative**, never a Q&A list:

- **Student A** — macroeconomic / geopolitical data specialist; later:
  forward + backward HMM algorithms; bull-regime examples; training-set
  rationale.
- **Student B** — microeconomic data specialist; later: Viterbi algorithm;
  bear-regime examples; validation-set rationale.
- **Student C** — financial/market data specialist; later: Baum-Welch
  algorithm; stagnant-regime examples; test-set rationale.

(If the real team has only 2 members, collapse roles per the "Groups of 2"
notes embedded in each task list below.)

## 4. Non-Negotiable Deliverable Format (per GWP, ×3)

Each GWP must ultimately produce, exactly matching the course's submission
rules:

1. **One LaTeX-built PDF report** (prose only, no code) — compiled from a
   proper `report`/`article` LaTeX project with a title page, section
   structure mirroring the task list, inline figures (`.pdf`/`.png` from the
   analysis), and a **bibliography in MLA format** (BibLaTeX with
   `style=mla` or `natbib` + a manually MLA-formatted `.bib`/`.bbl`).
2. **One executable Jupyter notebook** per GWP that:
   - imports reusable logic from the installable `src/` package (no
     copy-pasted logic duplicated inline),
   - is organized by explicit question/step numbers matching the PDF
     (e.g., `## Step 3 — ...`),
   - shows code, its real executed output (plots, tables), and a short
     interpretation cell after each result,
   - is fully re-run top-to-bottom with no errors before being exported.
3. **One exported HTML or PDF duplicate** of the executed notebook.

Grading rubric weighting to keep in mind while writing (40/30/30):
quantitative analysis & reasoning (40), technical+non-technical report
quality (30), writing/formatting/citations (20) — plus implicit
professionalism criteria (axes/labels/scales on every plot, no typos,
well-structured document).

## 5. Repository Engineering Standards

- **Language/paradigm**: Python 3.11+, strict **OOP** — no top-level
  "script soup". Each responsibility is a class with a single purpose:
  - `data/loaders.py`: `FredLoader`, `EiaLoader`, `YahooFinanceLoader`
    (all subclass an abstract `BaseDataLoader` with `.fetch() -> pd.DataFrame`).
  - `data/cleaners.py`: `OutlierCleaner`, `BadDataCleaner`,
    `MissingValueImputer`, composed by a `DataCleaningPipeline`.
  - `data/registry.py`: `DataDictionary` class that builds the
    variable/frequency/source/date-range metadata table.
  - `eda/plots.py`: `DistributionPlotter`, `TimeSeriesPlotter`,
    `MultivariatePlotter`.
  - `models/hmm.py`: `HiddenMarkovRegimeModel` wrapping forward/backward/
    Viterbi/Baum-Welch.
  - `models/bayesian_network.py`: `OilPriceBayesianNetwork` wrapping pgmpy
    structure (Hill-Climbing) + parameter learning/validation.
  - `models/black_litterman.py`: `BlackLittermanAllocator`.
  - `evaluation/metrics.py`: `ForecastEvaluator`, `DatasetSplitter`
    (train/validation/test).
  - Favor composition over inheritance beyond the loader/cleaner base
    classes; use `@dataclass` for simple config/value objects; use
    `abc.ABC` for the base classes above.
- **Packaging**: installable `src/`-layout package (`pyproject.toml`,
  editable install `pip install -e .`), semantic module names, type hints
  everywhere, Google-style or NumPy-style docstrings on every public method.
- **Testing**: `pytest` unit tests for every non-trivial class
  (cleaners, loaders with mocked HTTP, HMM toy example, BN structure
  learning on synthetic data). Target meaningful coverage, not 100%.
- **Quality gates**: `black` + `ruff` (or `flake8`) configured in
  `pyproject.toml`; optionally a `pre-commit` config.
- **Repo hygiene**: `.gitignore` (data caches, `.venv`, notebook
  checkpoints, LaTeX build artifacts), `README.md` with setup/run
  instructions, `LICENSE`, pinned `requirements.txt`, no secrets committed
  (API keys via `.env` + `python-dotenv`, with `.env.example` checked in).
- **Git workflow**: one commit per logical unit of work with descriptive
  messages; suggested branches `gwp1-data`, `gwp2-methodology`,
  `gwp3-results` merged into `main` as each phase completes.

## 6. Data Sourcing (GWP1, Step 3)

Collect macroeconomic, microeconomic/geopolitical, and financial variables
plausibly affecting the global price of oil, documenting source + frequency
+ date range for each in the data dictionary:

- **Financial/market** (`yfinance`, no key): WTI/Brent futures (`CL=F`,
  `BZ=F`), USD index (`DX-Y.NYB`), S&P 500, 10Y Treasury yield.
- **Macroeconomic** (`fredapi`/`pandas-datareader`, free FRED API key):
  US CPI, Fed funds rate, industrial production, US dollar index, global
  oil rig counts.
- **Geopolitical/energy** (`EIA Open Data API`, free API key): crude
  production, inventories (e.g., `PET.WCRSTUS1.W`), OPEC spare capacity if
  available.
- Skip SEDAR (Canadian filings) unless a specific microeconomic need
  arises — prefer EIA/FRED/Yahoo for reproducibility without manual
  scraping.

Store raw pulls under `data/raw/` (gitignored beyond a tiny sample),
cleaned under `data/processed/`, and cache a `data_dictionary.csv`.

## 7. Phased Execution Plan

Work **strictly in order** GWP1 → GWP2 → GWP3; do not start the next
phase's modeling code until the previous phase's data/report are complete,
since each stage's notebook imports artifacts from the previous one.

### Phase 1 — GWP1 (Problem Formulation & Data)
1. Write the 2–3 page problem formulation (what problem, why Bayesian
   networks fit, advantages) in `reports/GWP1/sections/problem_formulation.tex`.
2. Implement loaders + fetch all three data groups; build the data
   dictionary table.
3. Implement the 3 cleaning classes (outliers / bad data / missing value
   imputation) + the combined `DataCleaningPipeline`; document exclusions.
4. Implement EDA plotters (distributional, time series, multivariate);
   answer the 4 stylized-facts questions in prose.
5. Write the two-page PGM/parameter-vs-structure-learning/Markov
   chain-and-blanket summaries (one per student role) as report subsections.
6. Write the group pseudocode for "Algorithm 1: Inferred Causality".
7. Assemble `notebooks/GWP1_problem_formulation_and_data.ipynb` and
   `reports/GWP1/main.tex` (+ `references.bib` in MLA style). Export PDF/HTML.

### Phase 2 — GWP2 (Methodology & Model Development)
1. Implement `HiddenMarkovRegimeModel` with forward, backward, Viterbi,
   Baum-Welch methods; write pseudocode + toy examples per algorithm.
2. Detect/illustrate bull/bear/stagnant regimes on the cleaned oil series.
3. Write the team's own version of the HMM definition (Alvi §2.1.2.1) with
   a different supporting reference, ≤1–2 pages.
4. Write the 3 design subsections (data/indicator pipeline; regime process
   & `hmms` internals; `pgmpy` network training/validation) and merge into
   one 10–15 page narrative.
5. Document the regime-detection pipeline (§4.2–4.2.5 equivalent): series
   transformation, Baum-Welch parameter learning (black-box OK), Viterbi
   decoding (black-box OK); assign latent meaning to each hidden state.
6. Implement the Hill-Climbing minimal working example with `pgmpy`
   (`OilPriceBayesianNetwork`).
7. Assemble `notebooks/GWP2_methodology_and_model_development.ipynb` and
   `reports/GWP2/main.tex`. Export PDF/HTML.

### Phase 3 — GWP3 (Results & Interpretation)
1. Implement `DatasetSplitter` and document train/validation/test purposes
   (1–2 pages each) + a combined comparison/allocation section (≤6 pages).
2. Re-run Hill-Climbing Bayesian network end-to-end; assess replication of
   Alvi's results.
3. Implement `ForecastEvaluator`; report crude-oil forecast accuracy with a
   clear graphical display of predicted vs. actual.
4. Critically assess Alvi's 8 proposed contributions: cite specific
   pages/figures, judge whether each was achieved, and state why it matters.
5. Write the non-technical "Discussion" section focused on investment
   decisions/cost-effectiveness, avoiding model/algorithm jargon.
6. Assemble `notebooks/GWP3_results_and_interpretation.ipynb` and
   `reports/GWP3/main.tex`. Export PDF/HTML.

## 8. Definition of Done (checklist to self-verify before declaring a phase complete)

- [ ] All PDF task-list items for that GWP are addressed and traceable to a
      report section or notebook step.
- [ ] Every plot has axes labels, units, legend, and a caption.
- [ ] Every claim of fact has a citation; bibliography compiles in MLA
      style with no `biblatex`/`bibtex` warnings.
- [ ] `pytest` passes; `ruff`/`black` clean.
- [ ] Notebook runs top-to-bottom from a clean kernel with no errors.
- [ ] LaTeX report compiles to PDF with no errors (warnings about
      overfull boxes are acceptable but should be minimized).
- [ ] No secrets/API keys committed; `.env.example` documents required vars.
- [ ] Git history for the phase is clean and the working tree is clean.

## 9. Open Items Requiring Human Input

- Real names (or anonymized placeholders) and emails for the report
  template title page.
- Preferred GitHub repository name/visibility and whether to create it now
  or just prepare a local repo for the user to push.
- FRED/EIA API keys (free signup) — or confirm it's fine to proceed with
  `yfinance`-only financial data plus FRED's keyless CSV endpoints as a
  fallback.
- Confirm LaTeX build path: install a local TeX engine (e.g., via
  `winget install --id Tectonic.Tectonic` or MiKTeX) vs. compiling on
  Overleaf.
