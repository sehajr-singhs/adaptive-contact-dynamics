# Adaptive Contact-Rich Manipulation Under Changing Physics

**Project site:** https://sehajr-singhs.github.io/adaptive-contact-dynamics · **Paper:** [whitepaper.pdf](paper/whitepaper.pdf)

Learned manipulation policies fit the dynamics they trained on and break the moment the robot picks up a heavier object, the table gets slippery, or a motor heats up, and the field's usual fixes, more data and domain randomization and online fine-tuning, all help empirically without ever guaranteeing the system stays stable while it adapts. Classical Model Reference Adaptive Control gives exactly that guarantee through a Lyapunov-derived update law, but it assumes a clean analytic regressor that contact-rich manipulation does not hand you. This repo keeps the guarantee and still represents messy nonlinear uncertainty by putting a bounded radial-basis neural network inside the Lyapunov-derived law, so the network supplies the features and the law supplies the stability, and it tests one thing across four simulated environments, whether the controller recovers task performance after the contact physics drift mid-episode without losing stability.

The headline, all simulation-only and CPU-scale, five seeds, mean over seeds:

- On a gravity-loaded planar arm whose payload jumps 4x mid-episode, the constrained controller returns tracking error to **4.4 deg** where the fixed baseline stays at **27.2** and never recovers.
- On the real OGBench cube-single benchmark, driving the UR5e to push the cube to its goal under changing table friction and mass, it halves the arm's joint tracking error to **2.1 deg** and lands the cube within **2.8 cm** where the baseline degrades to **4.7 deg / 6.5 cm** and the matched-gain ablation goes fully unstable.
- The unconstrained ablation is identical except the stability machinery is removed, and the gain sweep shows why it matters, it is fine at low gain but has a stability ceiling above which it destabilizes, while the constrained law is stable at every gain, so the constraint removes the speed-versus-stability tradeoff.
- A separately trained neural Lyapunov certificate satisfies the exponential-decrease condition on **90.5%** of sampled states outside the ultimate bound.

The contribution is not a new architecture, it is the demonstration that the Lyapunov constraint lets you run the adaptation gain high enough to recover fast without the instability the same gain produces without it. The honest counterweight is in the table below and in the paper, Reacher is a benign over-actuated environment where adaptation buys nothing, and on the near-linear push plant the unconstrained law is also stable and slightly better, both of which we report straight because the credibility of the whole artifact rests on the table being honest.

## Results

| Environment | Controller | Post-drift error (step) | Recovered | Instability |
|---|---|---|---|---|
| E1 planar arm | Fixed baseline | 27.19 ± 0.15 deg | 0/5 | 0.00 |
| E1 planar arm | **Constrained (ours)** | **4.42 ± 0.04 deg** | **5/5** | **0.00** |
| E1 planar arm | Unconstrained (ablation) | 18.83 ± 4.46 deg | 0/5 | 0.53 |
| E2 reacher | Fixed baseline | 2.06 ± 0.00 deg | 5/5 | 0.00 |
| E2 reacher | Constrained (ours) | 3.40 ± 0.61 deg | 5/5 | 0.00 |
| E2 reacher | Unconstrained (ablation) | 7.25 ± 1.66 deg | 5/5 | 0.65 |
| E3 push | Fixed baseline | 3.45 ± 0.00 cm | 0/5 | 0.00 |
| E3 push | **Constrained (ours)** | **1.05 ± 0.03 cm** | **5/5** | 0.00 |
| E3 push | Unconstrained (ablation) | 0.44 ± 0.12 cm | 5/5 | 0.00 |
| E4 OGBench cube (arm) | Fixed baseline | 4.72 ± 0.09 deg | 0/5 | 0.00 |
| E4 OGBench cube (arm) | **Constrained (ours)** | **2.06 ± 0.74 deg** | **5/5** | **0.00** |
| E4 OGBench cube (arm) | Unconstrained (ablation) | 98.98 ± 21.28 deg | 0/5 | 1.00 |

Every number traces to a per-seed JSON under `results/` that a script wrote. The paper is in `paper/whitepaper.md` (render with `make paper`), and the website is `website/index.html`.

## Reproduce

```bash
pip install -r requirements.txt      # CPU-only; versions pinned and verified on Python 3.14
make quick                           # smoke-test every environment + figures, minutes on CPU
make full                            # the paper numbers: all seeds, both drift regimes, certificate,
                                     # gain sweeps, figures, master table, clips
make figures                         # regenerate every figure from saved JSON (no re-simulation)
make clips                           # re-render rollout clips from saved trajectories
make paper                           # render the whitepaper
```

`make quick` overwrites `results/` with a one-seed smoke test, so re-run `make full` to restore the paper numbers. A clean clone plus `make quick` reproduces the full pipeline end to end on CPU in minutes.

E4 is the real OGBench cube-single benchmark, whose `ogbench`/`dm_control`/`labmaze` dependencies do not build on Python 3.14, so it runs in a separate Python 3.11/3.12 environment:

```bash
py -3.12 -m venv .venv-ogbench
./.venv-ogbench/Scripts/python -m pip install ogbench pyyaml matplotlib imageio imageio-ffmpeg pillow
./.venv-ogbench/Scripts/python -m experiments.run_e4_ogbench      # runs the three controllers on real cube-single
./.venv-ogbench/Scripts/python -m experiments.make_clips_ogbench  # renders E4 clips from the real scene
```

The per-seed JSON it writes is the same schema as every other environment, so `make figures` and the master table pick it up automatically from the main environment.

## Repo layout

```
src/adaptive/      MRAC backbone, bounded RBF feature map, Lyapunov update, unconstrained ablation, certificate
src/controllers/   computed-torque PD baseline
src/envs/          the four MuJoCo environments + the shared drift-injection wrapper
src/metrics/       per-seed JSON logging, recovery and stability metrics
src/viz/           paper-quality plots, MuJoCo rollout rendering, architecture diagram
experiments/       run_all (quick/full), run_env, make_figures, make_clips, run_certificate, make_summary
configs/           one YAML per environment + the shared drift schedule
paper/             whitepaper.md (source of truth) and figures
website/           single static page, deploys to GitHub Pages with no build step
```

## What this is and is not

This is a research artifact, simulation-only, no hardware and no tactile sensing, with the contact in E3 modeled as Coulomb joint friction and the OGBench cube pushed to its goal by the real UR5e arm (six joints driven by computed torque, gripper closed as a pusher) rather than grasped and carried, because the real Robotiq grasp cannot dynamically carry the cube. The guarantee covers matched uncertainty, the part of the dynamics that enters through the control channel, and the certificate is verified by sampling over a bounded region rather than proven globally. E4 runs on the real OGBench `cube-single-play-singletask-v0` benchmark (ogbench 1.2.1, dm_control 1.0.43) in a separate Python 3.12 environment (`.venv-ogbench`), since those dependencies do not build on Python 3.14; the constrained controller halves the arm's joint tracking error and lands the cube more than twice as close to the goal, a real advantage but smaller than the seven-fold gap on an earlier self-contained stand-in, because the real cube is light relative to the heavy UR5e and the push is imperfect, and we report the smaller real numbers. The arm-in-the-loop task also makes the unconstrained ablation go unstable far more violently than a disembodied force does, which we report straight. The next step is to put this adaptive layer beneath a learned policy on a real arm with a tactile-sensed grasp and measure whether the guarantee holds when the contact is sensed rather than simulated.

## License

MIT.
