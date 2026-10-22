#pragma once

// UDP protocol with train.py / test.py (see agent/client.py for the Python side).
//
//   Python -> sim   [-100, -100, difficulty?]        start 25-body training mode
//                   [-101, -101, stress?]            start 1-body solo test mode (stress > 0 = auto pushes)
//                   10 floats per body (+ difficulty) set joint targets and advance 1/60 s
//   sim -> Python   per body: 48 state floats + 1 done flag (1.0 = fell and was respawned)

#include <algorithm>
#include <cmath>
#include <vector>
#include "platform/Engine.h"
#include "platform/Network.h"
#include "World.h"
#include "Disturbances.h"

constexpr int ACTIONS_PER_BODY = 10;
constexpr int STATE_FLOATS = 48;

// Packs the states + done flags of the first numToPack bodies into one packet.
// With checkDone, bodies that fell (or blew up) are respawned and flagged.
inline std::vector<float> packState(int numToPack, bool checkDone)
{
    std::vector<float> fullState;
    fullState.reserve(numToPack * (STATE_FLOATS + 1));

    for (int i = 0; i < numToPack; i++) {
        bool isDone = false;
        if (checkDone) {
            // A blown-up simulation (NaN/inf/flung away) must reset too: every comparison below
            // is false for NaN, so it would otherwise never be detected.
            bool unstable = !skeletons[i].isStable();

            // Standing hip is at ~161px, chest at ~216px, knees at ~81px (floor at 20px).
            // Reset if hip collapses (<70px), chest hits floor (<70px), OR torso folds into V-shape (chest drops near/below hip)
            bool hipFell   = (skeletons[i].bones[8].position.y < 70.0f);
            bool chestFell = (skeletons[i].bones[9].position.y < 70.0f);
            bool foldedV   = (skeletons[i].bones[9].position.y < skeletons[i].bones[8].position.y + 15.0f);
            isDone = unstable || hipFell || chestFell || foldedV;

            if (isDone) {
                // Respawn upright so the NEXT step starts clean
                // In solo test mode (numToPack == 1): always respawn with wild randomized arm/limb poses!
                // In training mode (numToPack > 1): mix random limbs so AI learns to balance from any pose
                bool randomArms = (numToPack == 1) || (rand() % 2 == 0);
                skeletons[i].init(spawnX[i], true, randomArms);
                disturbers[i].active = false;
            }
        }

        // Capture state (if it fell, sends the fresh standing state for the next action)
        std::vector<float> s = skeletons[i].getState();
        fullState.insert(fullState.end(), s.begin(), s.end());
        fullState.push_back(isDone ? 1.0f : 0.0f);
    }
    return fullState;
}

// Handles the two mode handshakes. Returns true if `actions` was a handshake (and a reply was sent).
inline bool handleHandshake(GLFWwindow* window, NetworkData& net, const std::vector<float>& actions)
{
    if (actions.size() < 2 || actions[0] != actions[1]) return false;

    // [-100.0, -100.0, difficulty?]: 25-Body Training Mode (train.py)
    if (actions[0] == -100.0f) {
        activeBodies = MAX_BODIES;
        autoDisturb = true;
        difficulty = (actions.size() >= 3 && std::isfinite(actions[2])) ? std::min(1.0f, std::max(0.0f, actions[2])) : 1.0f;
        glfwSetWindowTitle(window, "Standing Ragdoll AI - 25-Body Training Mode (Random Shoves Active)");
        for (int i = 0; i < MAX_BODIES; i++) {
            spawnX[i] = trainingSpawnX(i);
            skeletons[i].init(spawnX[i], true, (rand() % 2 == 0));
            disturbers[i].active = false;
        }
        net.send_state(packState(activeBodies, false));
        return true;
    }

    // [-101.0, -101.0, stress?]: 1-Body Solo Test Mode (test.py)
    // An optional third float > 0 turns on automatic shoves + virtual mouse balls at that difficulty.
    if (actions[0] == -101.0f) {
        activeBodies = 1;
        float stress = (actions.size() >= 3 && std::isfinite(actions[2])) ? actions[2] : 0.0f;
        autoDisturb = stress > 0.0f;
        difficulty = std::min(1.0f, std::max(0.0f, stress));
        glfwSetWindowTitle(window, "Standing Ragdoll AI - Solo Test Mode (Press 'R' for Random Pose | Cursor Ball | Spacebar)");
        spawnX[0] = 400.0f;
        skeletons[0].init(spawnX[0], true, true);
        disturbers[0].active = false;
        net.send_state(packState(activeBodies, false));
        return true;
    }

    return false;
}

// Is this a step packet: exactly 10 floats per active body, optionally +1 trailing difficulty float?
inline bool isStepPacket(const std::vector<float>& actions)
{
    size_t n = (size_t)(activeBodies * ACTIONS_PER_BODY);
    return actions.size() == n || actions.size() == n + 1;
}

// Applies a step packet's joint targets (and difficulty, in training mode) to the bodies.
inline void applyStepPacket(const std::vector<float>& actions)
{
    size_t n = (size_t)(activeBodies * ACTIONS_PER_BODY);
    if (actions.size() == n + 1 && activeBodies > 1 && std::isfinite(actions.back()))
        difficulty = std::min(1.0f, std::max(0.0f, actions.back()));

    for (int i = 0; i < activeBodies; i++) {
        std::vector<float> bodyActions(actions.begin() + i * ACTIONS_PER_BODY, actions.begin() + (i + 1) * ACTIONS_PER_BODY);
        skeletons[i].setActions(bodyActions);
    }
}
