#pragma once

#include <cmath>
#include "physics/Skeleton.h"

// Interactive cursor ball collision with ragdoll bones. Used for both the real mouse cursor and
// the virtual balls fired during training / stress tests.
inline void interactCursorBall(Skeleton& skel, vec2 mPos, vec2 mVel, float ballRadius)
{
    for (Bone& b : skel.bones) {
        float x1, y1, x2, y2;
        b.getEndpoints(x1, y1, x2, y2);

        vec2 A(x1, y1);
        vec2 B(x2, y2);
        vec2 AB = B - A;
        float lenSq = AB.x * AB.x + AB.y * AB.y;
        float t = 0.0f;
        if (lenSq > 1e-4f) {
            vec2 AP = mPos - A;
            t = (AP.x * AB.x + AP.y * AB.y) / lenSq;
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
        }

        vec2 closest = A + AB * t;
        vec2 diff = closest - mPos; // from cursor ball to bone
        float dist = sqrtf(diff.x * diff.x + diff.y * diff.y);
        float minDist = ballRadius + b.radius;

        if (dist < minDist) {
            vec2 norm = (dist > 1e-4f) ? (diff * (1.0f / dist)) : vec2(0.0f, 1.0f);
            float overlap = minDist - dist;

            // Push bone position away from cursor ball
            b.position = b.position + norm * overlap;

            // Apply push velocity from cursor movement and overlap penetration
            float pushStrength = overlap * 20.0f;
            b.velocity = b.velocity + norm * pushStrength + mVel * 0.35f;

            // Angular impulse from off-center contact
            vec2 r = closest - b.position;
            float torqueImpulse = (r.x * norm.y - r.y * norm.x) * pushStrength * 0.3f;
            b.angularVelocity += torqueImpulse / b.getInertia();
        }
    }
}
