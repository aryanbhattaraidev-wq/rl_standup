# rl_standup

A 2D ragdoll that learns to stand up and keep its balance, even when you shove it around with the mouse.

- **Simulation** (`sim/`): a hand-written C++ physics engine with OpenGL rendering. The ragdoll has 11 bones and 10 motorised joints.
- **Agent** (`agent/`): a PPO agent written in PyTorch.
- The two talk over UDP on localhost, in lockstep. Python sends 10 target joint angles per body, the sim advances one 1/60 s step and replies with the new state.

Windows only for now (Winsock, MinGW build).

## Requirements

- Windows with a MinGW-w64 `g++` on your `PATH` (for example [WinLibs](https://winlibs.com/))
- Python 3.10+ and PyTorch (`pip install -r requirements.txt`)
- GLFW 3 is vendored in `third_party/GLFW`

## Build the simulation

```powershell
.\build.ps1
```

This produces `rl_standup.exe`. The script builds with `-O2`; without it the sim is far too slow for training.

## Train

Use two terminals. Start the simulation first, then the trainer.

```powershell
.\rl_standup.exe
python train.py --steps 300000
```

- Training runs 25 ragdolls in parallel and takes roughly 35 to 40 minutes.
- Random shoves and flying "mouse balls" hit the ragdolls during training. Their difficulty ramps up over the first 40% of the run.
- Ctrl+C saves and stops. Add `--resume` to continue.
- `standup_policy.pth` is the latest policy, `standup_policy_best.pth` the best one seen at full difficulty, and `training_log.csv` the progress log. Watch `falls_per_1k_body_steps` go down.
- Run `python train.py --help` for the other options.

## Test

Close the training window, then start a fresh simulation.

```powershell
.\rl_standup.exe
python test.py
```

In the window: move the mouse to push the ragdoll with the orange ball, left-click and drag a limb, press **Space** for a shove, **R** for a new random pose.

| Flag | What it does |
| --- | --- |
| `--best` | use `standup_policy_best.pth` |
| `--model PATH` | use a specific checkpoint |
| `--stress D` | automatic pushes at difficulty `D` (0 to 1) |
| `--baseline` | ignore weights, just hold the standing pose |
| `--duration S --fast` | run `S` simulated seconds at full speed and print a summary (benchmarking) |

## Configuration

The ports default to 5005 (simulation) and 5006 (Python). Set `STANDUP_SIM_PORT` and `STANDUP_PY_PORT` before starting both sides to run a second instance alongside. `STANDUP_KP` and `STANDUP_KD` override the joint motor gains for experiments.

## Project layout

```
train.py, test.py        entry points (thin wrappers around agent/)
build.ps1                builds rl_standup.exe
agent/                   Python: the learning side
  config.py                constants shared with the sim (ports, dimensions, joint limits, file names)
  networks.py              observation scaling, policy (actor) and critic
  client.py                UDP client for the simulation
  reward.py                reward function
  ppo.py                   GAE and the clipped PPO update
  checkpoint.py            atomic saving and resuming
  train_loop.py            rollouts, logging, curriculum (python train.py)
  evaluate.py              evaluation and stress testing (python test.py)
sim/                     C++: the simulation side
  main.cpp                 main loop
  World.h                  shared state and constants
  Protocol.h               UDP handshakes, step packets, state packing, fall detection
  Simulation.h             lockstep and freeplay physics stepping
  Disturbances.h           random shoves and flying balls used in training / stress tests
  CursorBall.h             ball-vs-ragdoll collision
  Input.h                  mouse and keyboard callbacks
  Render.h                 OpenGL drawing
  physics/                 Vec2, Bone, Joint (motor + constraint), Skeleton (ragdoll)
  platform/                Engine (GLFW window), Network (Winsock UDP)
third_party/GLFW/        vendored GLFW 3 headers and library
docs/                    design sketch of the skeleton
```
