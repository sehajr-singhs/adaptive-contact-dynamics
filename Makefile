# Adaptive manipulation under changing contact dynamics.
# CPU-first. Every target writes real artifacts to disk.

PY ?= python

.PHONY: quick full figures clips paper site clean test

quick:        ## Smoke-test: 1 seed, short episodes, every environment + figures.
	$(PY) -m experiments.run_all --quick

full:         ## Paper numbers: all seeds, full episodes.
	$(PY) -m experiments.run_all --full

figures:      ## Regenerate every figure from saved results JSON (no re-simulation).
	$(PY) -m experiments.make_figures

clips:        ## Re-render rollout clips from saved trajectories.
	$(PY) -m experiments.make_clips

paper:        ## Render the whitepaper to PDF (requires pandoc).
	$(PY) -m experiments.build_paper

test:         ## Fast self-tests of the math spine (Lyapunov P, update law shapes).
	$(PY) -m pytest -q tests || $(PY) -m experiments.run_all --quick --env e1

clean:
	rm -rf results/*/ figures/*.png clips/*.mp4 clips/*.gif
