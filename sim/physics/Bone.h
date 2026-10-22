#pragma once

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#endif
#include <GL/gl.h>
#include <cmath>
#include "Vec2.h"

struct Bone
{
    vec2 position, velocity;
    float angle, halfLength, radius, angularVelocity, mass;

    Bone() : position(), velocity(), angle(0), halfLength(0), radius(0), angularVelocity(0), mass(1.0f) {}

    Bone(vec2 pos, vec2 vel, float ang, float halfLen, float rad)
        : position(pos), velocity(vel), angle(ang), halfLength(halfLen), radius(rad), angularVelocity(0), mass(1.0f) {}

    float getInertia() const
    {
        // Moment of inertia for a capsule / rod of length 2 * halfLength
        return (1.0f / 3.0f) * mass * (halfLength * halfLength + radius * radius);
    }

    void getEndpoints(float& x1, float& y1, float& x2, float& y2) const
    {
        float dx = cosf(angle) * halfLength;
        float dy = sinf(angle) * halfLength;
        x1 = position.x - dx; y1 = position.y - dy;
        x2 = position.x + dx; y2 = position.y + dy;
    }

    void drawCircle(float cx, float cy, float r, int segments = 20) const
    {
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(cx, cy);
        for (int i = 0; i <= segments; i++) {
            float theta = 2.0f * M_PI * (float)i / (float)segments;
            glVertex2f(cx + r * cosf(theta), cy + r * sinf(theta));
        }
        glEnd();
    }

    void draw() const
    {
        float x1, y1, x2, y2;
        getEndpoints(x1, y1, x2, y2);
        float px = -sinf(angle) * radius;
        float py = cosf(angle) * radius;
        glBegin(GL_QUADS);
        glVertex2f(x1 - px, y1 - py);
        glVertex2f(x1 + px, y1 + py);
        glVertex2f(x2 + px, y2 + py);
        glVertex2f(x2 - px, y2 - py);
        glEnd();
        drawCircle(x1, y1, radius);
        drawCircle(x2, y2, radius);
    }
};
