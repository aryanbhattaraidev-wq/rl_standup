#pragma once

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#endif
#include <GL/gl.h>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include "Vec2.h"
#include "Bone.h"

inline float envF(const char* name, float fallback)
{
    const char* v = getenv(name);
    return (v && *v) ? (float)atof(v) : fallback;
}

struct Joint
{
    int boneA;
    int boneB;
    vec2 localAnchorA;
    vec2 localAnchorB;
    float targetAngle; // desired relative angle between the two bones
    float maxTorque;   // maximum torque limit to prevent snapping limbs

    Joint(int bA, int bB, const std::vector<Bone>& bones, vec2 worldAnchor, float targetAng = 0.0f, float maxTrq = 800000.0f)
        : boneA(bA), boneB(bB), targetAngle(targetAng), maxTorque(maxTrq)
    {
        // Transform the world-space anchor point into the local space of each bone
        localAnchorA = rotate(worldAnchor - bones[boneA].position, -bones[boneA].angle);
        localAnchorB = rotate(worldAnchor - bones[boneB].position, -bones[boneB].angle);
    }

    vec2 getWorldAnchorA(const std::vector<Bone>& bones) const
    {
        return bones[boneA].position + rotate(localAnchorA, bones[boneA].angle);
    }

    vec2 getWorldAnchorB(const std::vector<Bone>& bones) const
    {
        return bones[boneB].position + rotate(localAnchorB, bones[boneB].angle);
    }

    // Applies a motor torque to pull the joint toward its targetAngle
    void applyTorque(std::vector<Bone>& bones, float dt)
    {
        Bone& a = bones[boneA];
        Bone& b = bones[boneB];

        // Current relative angle between boneB and boneA
        float currentAngle = b.angle - a.angle;

        // Difference between target angle and current angle
        float angleDiff = targetAngle - currentAngle;

        // Wrap to [-PI, PI] for the shortest rotation path
        while (angleDiff > M_PI) angleDiff -= 2.0f * M_PI;
        while (angleDiff < -M_PI) angleDiff += 2.0f * M_PI;

        // Relative angular velocity (damping term to prevent oscillation)
        float relAngVel = b.angularVelocity - a.angularVelocity;

        // Torque calculation: Spring stiffness * difference - damping * velocity
        // Stiffness must be high enough to carry the body's weight: at the old 420k the hips/knees
        // sagged ~0.7 rad under load and the untrained pose collapsed within a second. STANDUP_KP /
        // STANDUP_KD override these for experiments.
        static const float KP = envF("STANDUP_KP", 6000000.0f);
        static const float KD = envF("STANDUP_KD", 60000.0f);
        float invI = 1.0f / a.getInertia() + 1.0f / b.getInertia();
        // Both gains are capped by what the explicit integrator can handle for this pair of bones,
        // so light limbs (forearms, head) never jitter or blow up however stiff the heavy joints are.
        float kp = std::min(KP, 0.25f / (dt * dt * invI));
        float kd = std::min(KD, 0.7f / (dt * invI));
        float torque = kp * angleDiff - kd * relAngVel;

        // Clamp torque by maxTorque to prevent snapping
        if (torque > maxTorque) torque = maxTorque;
        if (torque < -maxTorque) torque = -maxTorque;

        // Angular impulse = torque * dt
        float angularImpulse = torque * dt;

        // Angular velocity change = angular impulse / moment of inertia
        a.angularVelocity -= angularImpulse / a.getInertia();
        b.angularVelocity += angularImpulse / b.getInertia();
    }

    // Solves position/anchor point constraint
    void solve(std::vector<Bone>& bones, float dt)
    {
        Bone& a = bones[boneA];
        Bone& b = bones[boneB];

        vec2 rA = rotate(localAnchorA, a.angle);
        vec2 rB = rotate(localAnchorB, b.angle);

        // Velocity of anchors = linear velocity + (angularVelocity x r)
        vec2 vA = a.velocity + vec2(-a.angularVelocity * rA.y, a.angularVelocity * rA.x);
        vec2 vB = b.velocity + vec2(-b.angularVelocity * rB.y, b.angularVelocity * rB.x);

        // Relative velocity between the two anchor points
        vec2 vRel = vA - vB;

        // Baumgarte stabilization (pushes anchors together to eliminate position drift)
        vec2 posA = a.position + rA;
        vec2 posB = b.position + rB;
        vec2 posError = posA - posB;
        float beta = 0.25f;
        vec2 vBias = posError * (beta / dt);

        // Target delta-velocity to cancel out relative speed difference + positional drift
        vec2 deltaV = (vRel + vBias) * -1.0f;

        // Effective mass matrix components (K = J * M^-1 * J^T)
        float invMassA = 1.0f / a.mass;
        float invMassB = 1.0f / b.mass;
        float invIA = 1.0f / a.getInertia();
        float invIB = 1.0f / b.getInertia();

        float k11 = invMassA + invMassB + rA.y * rA.y * invIA + rB.y * rB.y * invIB;
        float k22 = invMassA + invMassB + rA.x * rA.x * invIA + rB.x * rB.x * invIB;
        float k12 = -rA.x * rA.y * invIA - rB.x * rB.y * invIB;

        float det = k11 * k22 - k12 * k12;
        if (fabsf(det) < 1e-6f) return;

        // Compute impulse: impulse = K^-1 * deltaV
        vec2 impulse;
        impulse.x = (k22 * deltaV.x - k12 * deltaV.y) / det;
        impulse.y = (-k12 * deltaV.x + k11 * deltaV.y) / det;

        // Apply impulse to Bone A (linear = impulse / mass, angular = (r x impulse) / I)
        a.velocity = a.velocity + impulse * invMassA;
        a.angularVelocity += (rA.x * impulse.y - rA.y * impulse.x) * invIA;

        // Apply negative impulse to Bone B
        b.velocity = b.velocity - impulse * invMassB;
        b.angularVelocity -= (rB.x * impulse.y - rB.y * impulse.x) * invIB;
    }

    void draw(const std::vector<Bone>& bones) const
    {
        vec2 p = getWorldAnchorA(bones);
        float r = 4.0f;
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(p.x, p.y);
        for (int i = 0; i <= 16; i++) {
            float theta = 2.0f * M_PI * (float)i / 16.0f;
            glVertex2f(p.x + r * cosf(theta), p.y + r * sinf(theta));
        }
        glEnd();
    }
};
