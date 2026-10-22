#pragma once

#include "World.h"
#include "CursorBall.h"
#include "Disturbances.h"

constexpr int SUB_STEPS = 4;

// One lockstep tick: exactly 1/60 s of physics per network command from Python.
inline void stepLockstep()
{
    // Track mouse velocity
    mouseVel = (mousePos - prevMousePos) * 60.0f;
    prevMousePos = mousePos;

    if (autoDisturb) {
        for (int s = 0; s < activeBodies; s++) spawnDisturbances(s);
    }

    const float subDt = (1.0f / 60.0f) / (float)SUB_STEPS;
    for (int step = 0; step < SUB_STEPS; step++) {
        for (int s = 0; s < activeBodies; s++) {
            int dragIdx = (s == draggedSkeletonIndex) ? draggedBoneIndex : -1;
            skeletons[s].updatePhysics(subDt, dragIdx, mousePos);
        }

        // Interactive cursor ball collision for solo test ragdoll
        if (activeBodies == 1) {
            interactCursorBall(skeletons[0], mousePos, mouseVel, CURSOR_BALL_RADIUS);
        }
        if (autoDisturb) {
            for (int s = 0; s < activeBodies; s++) stepDisturber(s, subDt);
        }
    }
}

// Standalone freeplay (no Python connected): real-time physics so you can poke the ragdoll.
inline void stepFreeplay(float dt)
{
    if (dt > 0.033f) dt = 0.033f;

    mouseVel = (mousePos - prevMousePos) * 60.0f;
    prevMousePos = mousePos;

    const float subDt = dt / (float)SUB_STEPS;
    for (int step = 0; step < SUB_STEPS; step++) {
        for (int s = 0; s < activeBodies; s++) {
            int dragIdx = (s == draggedSkeletonIndex) ? draggedBoneIndex : -1;
            skeletons[s].updatePhysics(subDt, dragIdx, mousePos);
        }

        if (activeBodies == 1) {
            interactCursorBall(skeletons[0], mousePos, mouseVel, CURSOR_BALL_RADIUS);
        }
    }
}
