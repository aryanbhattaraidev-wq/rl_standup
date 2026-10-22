# rl_standup

A 2D ragdoll that learns to stand up and keep its balance, even when you shove it around with the mouse.

- **Simulation** (`rl_standup.cpp` and the `*.h` files): a hand-written C++ physics engine with OpenGL rendering. The ragdoll has 11 bones and 10 motorised joints.
- **Agent** (`train.py`, `test.py`): a PPO agent written in PyTorch.
- The two talk over UDP on localhost, in lockstep. Python sends 10 target joint angles per body, the sim advances one 1/60 s step and replies with the new state.

Windows only for now (Winsock, MinGW build).

## Requirements

- Windows with a MinGW-w64 `g++` (for example [WinLibs](https://winlibs.com/))
- Python 3.10+ and PyTorch (`pip install -r requirements.txt`)
- GLFW 3 is vendored in `GLFW/`

## Build the simulation

```powershell
g++ -O2 -std=c++17 rl_standup.cpp -o rl_standup.exe -I GLFW/include GLFW/lib/libglfw3.a -lopengl32 -lgdi32 -lws2_32
```

Keep `-O2`; without it the sim is far too slow for training.

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

## Layout

| File | Purpose |
| --- | --- |
| `rl_standup.cpp` | main loop, UDP handling, mouse input, automatic disturbances, rendering |
| `Skeleton.h`, `Bone.h`, `Joint.h` | ragdoll physics, motors and constraints |
| `Network.h`, `Engine.h`, `Vec2.h` | UDP sockets, GLFW window, vector maths |
| `train.py` | policy and critic networks, reward, PPO training, UDP client |
| `test.py` | evaluation and stress testing |
