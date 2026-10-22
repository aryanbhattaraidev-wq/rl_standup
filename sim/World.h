#pragma once

// Shared simulation state. Everything is `inline` (C++17) so the headers can share one definition.

#include <cstdlib>
#include <vector>
#include "physics/Skeleton.h"

constexpr int MAX_BODIES = 25;
constexpr float WORLD_W = 800.0f;
constexpr float WORLD_H = 600.0f;
constexpr float CURSOR_BALL_RADIUS = 16.0f;

inline std::vector<Skeleton> skeletons(MAX_BODIES);
inline std::vector<float> spawnX(MAX_BODIES);
inline int activeBodies = 1; // Default to 1 centered body for clean testing & standalone

// Mouse & cursor ball state
inline int draggedSkeletonIndex = -1;
inline int draggedBoneIndex = -1;
inline vec2 mousePos(400, 300);
inline vec2 prevMousePos(400, 300);
inline vec2 mouseVel(0, 0);

// Automatic disturbances. During training every body gets random shoves and "virtual mouse balls"
// that fly through it, so the policy sees the same kind of pushes a human gives with the real cursor.
// difficulty (0..1) is sent by Python as a trailing float so it can ramp up as the agent improves.
inline float difficulty = 1.0f;
inline bool autoDisturb = false; // always on in training mode; opt-in for solo test (stress test)

inline float frand() { return (float)rand() / (float)RAND_MAX; }

inline int envInt(const char* name, int fallback)
{
    const char* v = getenv(name);
    return (v && *v) ? atoi(v) : fallback;
}

// Spawn position of body i in the 25-body training grid
inline float trainingSpawnX(int i) { return 120.0f + (i % 5) * 140.0f + ((i / 5) - 2) * 8.0f; }
