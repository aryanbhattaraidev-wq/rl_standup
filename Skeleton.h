#pragma once

#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "Vec2.h"
#include "Bone.h"
#include "Joint.h"

struct Skeleton
{
    std::vector<Bone> bones;
    std::vector<Joint> joints;
    float spawnX = 400.0f;

    void init(float sx = 400.0f, bool randomize = false, bool randomLimbs = false)
    {
        spawnX = sx;
        float dx = spawnX - 400.0f;

        auto rNoise = [randomize](float scale = 0.12f) -> float {
            if (!randomize) return 0.0f;
            return ((float)rand() / (float)RAND_MAX - 0.5f) * scale;
        };

        // Torso, head, and leg angles
        float hipAng   = 0.0f + rNoise(0.10f);
        float chestAng = 3.14159f / 2.0f + rNoise(0.15f);
        float headAng  = 3.14159f / 2.0f + rNoise(0.20f);
        float legLAng  = 3.14159f / 2.3f + rNoise(0.06f);
        float legRAng  = -3.14159f / 2.3f + rNoise(0.06f);
        float calfLAng = 3.14159f / 2.0f + rNoise(0.05f);
        float calfRAng = 3.14159f / 2.0f + rNoise(0.05f);

        // Arms and forearms: wild randomized poses when randomLimbs is true
        float armLAng = 0.67f;
        float armRAng = -0.67f;
        float forearmLAng = 1.05f;
        float forearmRAng = -1.05f;

        if (randomLimbs) {
            // Left arm & forearm: wide range (hands up: -1.6, hands out: 0.0, hands down: +1.6)
            armLAng = ((float)rand() / (float)RAND_MAX) * 3.2f - 1.6f;
            forearmLAng = armLAng + (((float)rand() / (float)RAND_MAX) * 1.8f - 0.9f);

            // Right arm & forearm: wide range (hands up: +1.6, hands out: 0.0, hands down: -1.6)
            armRAng = ((float)rand() / (float)RAND_MAX) * 3.2f - 1.6f;
            forearmRAng = armRAng + (((float)rand() / (float)RAND_MAX) * 1.8f - 0.9f);
        } else {
            armLAng += rNoise(0.12f);
            armRAng += rNoise(0.12f);
            forearmLAng += rNoise(0.12f);
            forearmRAng += rNoise(0.12f);
        }

        // Exact shoulder anchors on chest
        vec2 shoulderL(380.0f + dx, 231.0f);
        vec2 shoulderR(420.0f + dx, 231.0f);

        // Compute forward kinematics for arms so joints are perfectly aligned
        vec2 armLPos = shoulderL - vec2(cosf(armLAng), sinf(armLAng)) * 20.0f;
        vec2 elbowL  = shoulderL - vec2(cosf(armLAng), sinf(armLAng)) * 40.0f;
        vec2 forearmLPos = elbowL - vec2(cosf(forearmLAng), sinf(forearmLAng)) * 18.0f;

        vec2 armRPos = shoulderR + vec2(cosf(armRAng), sinf(armRAng)) * 20.0f;
        vec2 elbowR  = shoulderR + vec2(cosf(armRAng), sinf(armRAng)) * 40.0f;
        vec2 forearmRPos = elbowR + vec2(cosf(forearmRAng), sinf(forearmRAng)) * 18.0f;

        // 1. Initialize Bones (shifted so feet rest directly on floor at y = 20)
        bones = {
            Bone(armRPos, vec2(0), armRAng, 20, 7),               // 0: armR
            Bone(armLPos, vec2(0), armLAng, 20, 7),               // 1: armL
            Bone(forearmRPos, vec2(0), forearmRAng, 18, 6),       // 2: forearmR
            Bone(forearmLPos, vec2(0), forearmLAng, 18, 6),       // 3: forearmL
            Bone(vec2(375 + dx, 121), vec2(0), legLAng, 28, 8),   // 4: legL
            Bone(vec2(425 + dx, 121), vec2(0), legRAng, 28, 8),   // 5: legR
            Bone(vec2(365 + dx, 51),  vec2(0), calfLAng, 24, 7),  // 6: calfL (feet at y = 20)
            Bone(vec2(435 + dx, 51),  vec2(0), calfRAng, 24, 7),  // 7: calfR (feet at y = 20)
            Bone(vec2(400 + dx, 161), vec2(0), hipAng, 16, 10),   // 8: hip/torso
            Bone(vec2(400 + dx, 216), vec2(0), chestAng, 28, 15), // 9: chest/body
            Bone(vec2(400 + dx, 281), vec2(0), headAng, 12, 11),  // 10: head
        };

        // 1b. Set anatomically proportional masses
        bones[0].mass = 1.0f;   // armR
        bones[1].mass = 1.0f;   // armL
        bones[2].mass = 0.7f;   // forearmR
        bones[3].mass = 0.7f;   // forearmL
        bones[4].mass = 2.0f;   // legL (thigh)
        bones[5].mass = 2.0f;   // legR (thigh)
        bones[6].mass = 1.5f;   // calfL
        bones[7].mass = 1.5f;   // calfR
        bones[8].mass = 3.0f;   // hip/pelvis
        bones[9].mass = 5.0f;   // chest/torso (heaviest)
        bones[10].mass = 1.5f;  // head

        // 2. Initialize Joints
        joints.clear();
        auto addJoint = [this, dx](int bA, int bB, vec2 anchor, float maxTorque = 800000.0f) {
            float defaultTargetAngle = bones[bB].angle - bones[bA].angle;
            joints.push_back(Joint(bA, bB, bones, vec2(anchor.x + dx, anchor.y), defaultTargetAngle, maxTorque));
        };

        // Neck: Head (10) <-> Chest (9)
        addJoint(10, 9, vec2(400, 251), 1200000.0f);
        // Waist: Chest (9) <-> Hip (8)
        addJoint(9, 8, vec2(400, 181), 2000000.0f);
        // Left Shoulder: Chest (9) <-> ArmL (1)
        joints.push_back(Joint(9, 1, bones, shoulderL, bones[1].angle - bones[9].angle, 800000.0f));
        // Right Shoulder: Chest (9) <-> ArmR (0)
        joints.push_back(Joint(9, 0, bones, shoulderR, bones[0].angle - bones[9].angle, 800000.0f));
        // Left Elbow: ArmL (1) <-> ForearmL (3)
        joints.push_back(Joint(1, 3, bones, elbowL, bones[3].angle - bones[1].angle, 600000.0f));
        // Right Elbow: ArmR (0) <-> ForearmR (2)
        joints.push_back(Joint(0, 2, bones, elbowR, bones[2].angle - bones[0].angle, 600000.0f));
        // Left Hip: Hip (8) <-> LegL (4)
        addJoint(8, 4, vec2(385, 151), 1800000.0f);
        // Right Hip: Hip (8) <-> LegR (5)
        addJoint(8, 5, vec2(415, 151), 1800000.0f);
        // Left Knee: LegL (4) <-> CalfL (6)
        addJoint(4, 6, vec2(365, 81), 1400000.0f);
        // Right Knee: LegR (5) <-> CalfR (7)
        addJoint(5, 7, vec2(435, 81), 1400000.0f);
    }

    // False if the simulation blew up (NaN/inf or flung far outside the window)
    bool isStable() const
    {
        for (const Bone& b : bones) {
            if (!std::isfinite(b.position.x) || !std::isfinite(b.position.y) ||
                !std::isfinite(b.angle) || !std::isfinite(b.angularVelocity) ||
                !std::isfinite(b.velocity.x) || !std::isfinite(b.velocity.y))
                return false;
            if (fabsf(b.position.x) > 2000.0f || fabsf(b.position.y) > 2000.0f)
                return false;
        }
        return true;
    }

    void handleBorderCollisions(Bone& b)
    {
        float floorY = 20.0f;
        float leftX = 20.0f;
        float rightX = 780.0f;

        float x1, y1, x2, y2;
        b.getEndpoints(x1, y1, x2, y2);

        float minY = std::min(b.position.y - b.radius, std::min(y1 - b.radius, y2 - b.radius));
        if (minY < floorY) {
            float penetration = floorY - minY;
            b.position.y += penetration;
            if (b.velocity.y < 0) {
                b.velocity.y = -b.velocity.y * 0.1f; // low bounce
            }

            // Contact point on floor
            float contactX = (y1 < y2) ? x1 : x2;
            float contactY = (y1 < y2) ? y1 : y2;
            vec2 r = vec2(contactX, contactY) - b.position;

            // Horizontal velocity at contact point: v_contact = v_linear + omega x r
            float contactVx = b.velocity.x - b.angularVelocity * r.y;

            // Effective inverse mass for contact point impulse
            float invMassEff = (1.0f / b.mass) + (r.y * r.y) / b.getInertia();
            if (invMassEff > 1e-5f) {
                // Apply dynamic friction impulse at contact point
                float frictionImpulse = -contactVx * 0.75f / invMassEff;
                b.velocity.x += frictionImpulse / b.mass;
                b.angularVelocity -= (r.y * frictionImpulse) / b.getInertia();

                // Static friction lock: firmly pin foot contact to prevent outward split creeping
                float remainingVx = b.velocity.x - b.angularVelocity * r.y;
                if (fabsf(remainingVx) < 40.0f) {
                    float lockImpulse = -remainingVx / invMassEff;
                    b.velocity.x += lockImpulse / b.mass;
                    b.angularVelocity -= (r.y * lockImpulse) / b.getInertia();
                }
            }

            b.angularVelocity *= 0.85f;
        }

        if (b.position.x - b.radius < leftX) {
            b.position.x = leftX + b.radius;
            b.velocity.x = -b.velocity.x * 0.3f;
        }
        if (b.position.x + b.radius > rightX) {
            b.position.x = rightX - b.radius;
            b.velocity.x = -b.velocity.x * 0.3f;
        }
    }

    void updatePhysics(float dt, int draggedBoneIndex, vec2 mousePos)
    {
        float gravity = -2200.0f; // pixel-space gravity

        // 1. Apply gravity or mouse dragging force
        for (size_t i = 0; i < bones.size(); i++) {
            if ((int)i == draggedBoneIndex) {
                vec2 pull = (mousePos - bones[i].position) * 25.0f;
                bones[i].velocity = pull;
            } else {
                bones[i].velocity.y += gravity * dt;
            }
        }

        // 2. Apply motor torques toward targetAngle
        for (Joint& joint : joints) {
            joint.applyTorque(bones, dt);
        }

        // 3. Solve joint position/velocity constraints
        const int iterations = 10;
        for (int it = 0; it < iterations; it++) {
            for (Joint& joint : joints) {
                joint.solve(bones, dt);
            }
        }

        // 4. Integrate velocities into positions & apply damping
        for (Bone& b : bones) {
            b.position = b.position + b.velocity * dt;
            b.angle += b.angularVelocity * dt;

            b.velocity = b.velocity * 0.999f;
            b.angularVelocity *= 0.99f;

            // 5. Border collisions
            handleBorderCollisions(b);
        }
    }

    std::vector<float> getState() const
    {
        std::vector<float> state;
        state.reserve(48); // 11 bones × 3 + 4 hip + 4 footL + 4 footR + 3 chest = 48

        // 1. cos(angle), sin(angle), and NORMALIZED angular velocity for each bone (33 floats)
        for (const Bone& b : bones) {
            state.push_back(cosf(b.angle));
            state.push_back(sinf(b.angle));
            state.push_back(b.angularVelocity * 0.01f); // scale down to prevent network saturation
        }

        // 2. Hip (bone 8): position and velocity (4 floats) [indices 33-36]
        const Bone& hip = bones[8];
        state.push_back(hip.position.x - spawnX); // Relative horizontal position
        state.push_back(hip.position.y);
        state.push_back(hip.velocity.x);
        state.push_back(hip.velocity.y);

        // 3. Left foot endpoint — calf L bone 6 (4 floats) [indices 37-40]
        {
            float cx1, cy1, cx2, cy2;
            bones[6].getEndpoints(cx1, cy1, cx2, cy2);
            // Lower endpoint is the foot
            float footX, footY, rx, ry;
            if (cy1 < cy2) { footX = cx1; footY = cy1; rx = cx1 - bones[6].position.x; ry = cy1 - bones[6].position.y; }
            else            { footX = cx2; footY = cy2; rx = cx2 - bones[6].position.x; ry = cy2 - bones[6].position.y; }
            float footVx = bones[6].velocity.x - bones[6].angularVelocity * ry;
            float footVy = bones[6].velocity.y + bones[6].angularVelocity * rx;
            state.push_back(footX - spawnX);
            state.push_back(footY);
            state.push_back(footVx);
            state.push_back(footVy);
        }

        // 4. Right foot endpoint — calf R bone 7 (4 floats) [indices 41-44]
        {
            float cx1, cy1, cx2, cy2;
            bones[7].getEndpoints(cx1, cy1, cx2, cy2);
            float footX, footY, rx, ry;
            if (cy1 < cy2) { footX = cx1; footY = cy1; rx = cx1 - bones[7].position.x; ry = cy1 - bones[7].position.y; }
            else            { footX = cx2; footY = cy2; rx = cx2 - bones[7].position.x; ry = cy2 - bones[7].position.y; }
            float footVx = bones[7].velocity.x - bones[7].angularVelocity * ry;
            float footVy = bones[7].velocity.y + bones[7].angularVelocity * rx;
            state.push_back(footX - spawnX);
            state.push_back(footY);
            state.push_back(footVx);
            state.push_back(footVy);
        }

        // 5. Chest (bone 9): position and vertical velocity (3 floats) [indices 45-47]
        const Bone& chest = bones[9];
        state.push_back(chest.position.x - spawnX);
        state.push_back(chest.position.y);
        state.push_back(chest.velocity.y);

        return state;
    }

    void setActions(const std::vector<float>& actions)
    {
        // Joint limits [min, max] for 10 joints:
        // 0: Neck, 1: Waist, 2: ShoulderL, 3: ShoulderR, 4: ElbowL, 5: ElbowR,
        // 6: HipL, 7: HipR, 8: KneeL, 9: KneeR
        static const float jointMin[10] = {-0.8f, -2.5f, -2.8f, -3.2f, -0.3f, -2.6f,  0.4f, -2.4f, -0.2f,  1.2f};
        static const float jointMax[10] = { 0.8f, -0.6f,  1.2f, -0.8f,  2.6f,  0.3f,  2.4f, -0.4f,  2.5f,  3.3f};

        for (size_t i = 0; i < actions.size() && i < joints.size(); i++) {
            float a = actions[i];
            if (!std::isfinite(a)) continue; // ignore NaN/inf, keep previous target
            if (i < 10) {
                if (a < jointMin[i]) a = jointMin[i];
                if (a > jointMax[i]) a = jointMax[i];
            }
            joints[i].targetAngle = a;
        }
    }

    void draw(float r = 0.9f, float g = 0.85f, float b = 0.7f, float a = 1.0f) const
    {
        // Draw bones with alpha
        glColor4f(r, g, b, a);
        for (const Bone& bone : bones) {
            bone.draw();
        }

        // Draw joint pins with alpha
        glColor4f(0.2f, 0.2f, 0.2f, a);
        for (const Joint& joint : joints) {
            joint.draw(bones);
        }
    }
};
