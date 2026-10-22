#pragma once

#include "World.h"
#include "CursorBall.h"

// A virtual cursor ball flying through one body
struct Disturber
{
    bool active = false;
    vec2 pos;
    vec2 vel;
    int substepsLeft = 0;
};
inline std::vector<Disturber> disturbers(MAX_BODIES);

// Once per physics step: maybe launch a virtual mouse ball at body s and/or give it a random shove.
inline void spawnDisturbances(int s)
{
    const float scale = 0.3f + 0.7f * difficulty;
    Skeleton& skel = skeletons[s];

    // Virtual cursor ball sweeping horizontally through the body at a random height and speed.
    // Heights span feet to head and speeds span a gentle nudge to a hard flick of the mouse.
    Disturber& d = disturbers[s];
    if (!d.active && frand() < scale / 150.0f) {
        float side = (rand() % 2 == 0) ? 1.0f : -1.0f;
        float speed = 150.0f + frand() * (250.0f + 700.0f * difficulty);
        float y = 35.0f + frand() * 290.0f;
        d.pos = vec2(skel.bones[8].position.x - side * (160.0f + frand() * 60.0f), y);
        d.vel = vec2(side * speed, (frand() - 0.5f) * 200.0f);
        d.substepsLeft = (int)(420.0f / speed * 240.0f) + 1; // enough travel to pass fully through
        d.active = true;
    }

    // Instantaneous velocity shove on hip + chest (same as the Spacebar), at random strength
    if (frand() < scale / 90.0f) {
        float shoveX = (rand() % 2 == 0 ? 1.0f : -1.0f) * (150.0f + frand() * (150.0f + 600.0f * difficulty));
        skel.bones[8].velocity.x += shoveX;
        skel.bones[9].velocity.x += shoveX * 0.75f;
    }
}

// Once per physics substep: advance the virtual ball and let it collide with the body.
inline void stepDisturber(int s, float subDt)
{
    Disturber& d = disturbers[s];
    if (!d.active) return;
    d.pos = d.pos + d.vel * subDt;
    interactCursorBall(skeletons[s], d.pos, d.vel, CURSOR_BALL_RADIUS);
    if (--d.substepsLeft <= 0) d.active = false;
}
