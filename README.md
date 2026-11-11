# Brownian Motion and Measure Theory

## Project description

**Project type:** Student-selected independent study project  
**Course:** Stochastic Processes  
**Instructor:** Dr. Hossein Peyvandi  
**Prepared by:** Yasin Saifi

This project was undertaken as the final project for the course Stochastic Processes, with the instructor's approval. The selected topic provides an opportunity for focused study of measure-theoretic probability and standard Brownian motion, extending the foundational course material through mathematical exposition and computational investigation.

The study is intended to develop a systematic understanding of the mathematical framework of probability, the definition and properties of standard Brownian motion, and selected numerical illustrations. The report is organized to support individual study as well as presentation and discussion of the material in class.

## Project contents

- `report.pdf`: typeset mathematical and computational report.
- `report.tex`: complete LaTeX source.
- `src/brownian.py`: Brownian path simulation, quadratic variation, and time-integral approximation.
- `src/run_experiments.py`: script to regenerate the figures.
- `figures/`: figures generated from the simulations.

## Reproducing the experiments

```bash
python -m pip install -r src/requirements.txt
python src/run_experiments.py
```

The plots use fixed random seeds. The Brownian generator simulates independent increments with distribution `N(0, Δt)`. Piecewise-linear segments in the path plot connect grid values and provide a visualization of the simulated process on the selected time grid.
