#include "Engine.h"
#include "Skeleton.h"
#include "Network.h"
#include <vector>
#include <ctime>
#include <cstdlib>

const int MAX_BODIES = 25;
const float WORLD_W = 800.0f;
const float WORLD_H = 600.0f;
std::vector<Skeleton> skeletons(MAX_BODIES);
std::vector<float> spawnX(MAX_BODIES);
int activeBodies = 1; // Default to 1 centered body for clean testing & standalone

// Mouse & Cursor Ball state
int draggedSkeletonIndex = -1;
int draggedBoneIndex = -1;
vec2 mousePos(400, 300);
vec2 prevMousePos(400, 300);
vec2 mouseVel(0, 0);
const float CURSOR_BALL_RADIUS = 16.0f;

// Automatic disturbances. During training every body gets random shoves and "virtual mouse balls"
// that fly through it, so the policy sees the same kind of pushes a human gives with the real cursor.
// difficulty (0..1) is sent by Python as a trailing float so it can ramp up as the agent improves.
float difficulty = 1.0f;
bool autoDisturb = false; // always on in training mode; opt-in for solo test (stress test)

struct Disturber
{
    bool active = false;
    vec2 pos;
    vec2 vel;
    int substepsLeft = 0;
};
std::vector<Disturber> disturbers(MAX_BODIES);

float frand() { return (float)rand() / (float)RAND_MAX; }

int envInt(const char* name, int fallback)
{
    const char* v = getenv(name);
    return (v && *v) ? atoi(v) : fallback;
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            float minDist = 70.0f;
            draggedSkeletonIndex = -1;
            draggedBoneIndex = -1;
            for (int s = 0; s < activeBodies; s++) {
                for (size_t i = 0; i < skeletons[s].bones.size(); i++) {
                    vec2 diff = skeletons[s].bones[i].position - mousePos;
                    float dist = sqrtf(diff.x * diff.x + diff.y * diff.y);
                    if (dist < minDist) {
                        minDist = dist;
                        draggedSkeletonIndex = s;
                        draggedBoneIndex = (int)i;
                    }
                }
            }
        } else if (action == GLFW_RELEASE) {
            draggedSkeletonIndex = -1;
            draggedBoneIndex = -1;
        }
    }
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    // OpenGL ortho: (0, 0) is bottom-left, (800, 600) is top-right. Map from actual window size
    // so the cursor stays aligned if the window is resized or the display is scaled.
    int w = 1, h = 1;
    glfwGetWindowSize(window, &w, &h);
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    mousePos.x = (float)xpos * WORLD_W / (float)w;
    mousePos.y = WORLD_H - (float)ypos * WORLD_H / (float)h;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        // Spacebar delivers a sharp random shove to the solo test ragdoll to test balance!
        float shoveX = (rand() % 2 == 0 ? 1.0f : -1.0f) * (300.0f + frand() * 350.0f);
        skeletons[0].bones[8].velocity.x += shoveX;
        skeletons[0].bones[9].velocity.x += shoveX * 0.7f;
    }
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        // Press R to trigger a fresh random pose respawn anytime!
        skeletons[0].init(spawnX[0], true, true);
        disturbers[0].active = false;
    }
}

// Interactive cursor ball collision with ragdoll bones
void interactCursorBall(Skeleton& skel, vec2 mPos, vec2 mVel, float ballRadius)
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

// Once per physics step: maybe launch a virtual mouse ball at body s and/or give it a random shove.
void spawnDisturbances(int s)
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
void stepDisturber(int s, float subDt)
{
    Disturber& d = disturbers[s];
    if (!d.active) return;
    d.pos = d.pos + d.vel * subDt;
    interactCursorBall(skeletons[s], d.pos, d.vel, CURSOR_BALL_RADIUS);
    if (--d.substepsLeft <= 0) d.active = false;
}

void drawDisc(vec2 c, float r, int segs)
{
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(c.x, c.y);
    for (int seg = 0; seg <= segs; seg++) {
        float th = 2.0f * M_PI * (float)seg / (float)segs;
        glVertex2f(c.x + r * cosf(th), c.y + r * sinf(th));
    }
    glEnd();
}

int main(void)
{
    srand((unsigned)time(NULL));

    Engine engine(800, 600, "Standing Ragdoll AI");
    if (!engine.init())
        return -1;

    // Sockets: C++ listens on STANDUP_SIM_PORT (5005), sends to Python on STANDUP_PY_PORT (5006).
    // Override both (and the matching vars for train.py/test.py) to run a second instance alongside.
    NetworkData net("127.0.0.1", envInt("STANDUP_SIM_PORT", 5005), envInt("STANDUP_PY_PORT", 5006));

    glfwSetMouseButtonCallback(engine.window, mouseButtonCallback);
    glfwSetCursorPosCallback(engine.window, cursorPosCallback);
    glfwSetKeyCallback(engine.window, keyCallback);

    // Initial setup: Body 0 is centered at x = 400.0f with random limbs enabled
    spawnX[0] = 400.0f;
    skeletons[0].init(spawnX[0], true, true);

    for (int i = 1; i < MAX_BODIES; i++) {
        spawnX[i] = 120.0f + (i % 5) * 140.0f + ((i / 5) - 2) * 8.0f;
        skeletons[i].init(spawnX[i], true, (rand() % 2 == 0));
    }

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 800, 0, 600, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Enable smooth alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    double lastTime = glfwGetTime();

    // Helper lambda to pack active skeleton states + done flags into a single vector
    auto packState = [&](int numToPack, bool checkDone) {
        std::vector<float> fullState;
        fullState.reserve(numToPack * 49);

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
    };

    bool networkActive = false;
    unsigned long stepCounter = 0;

    while (!glfwWindowShouldClose(engine.window))
    {
        // 1. Process network commands from Python
        std::vector<float> actions;
        bool gotPacket = net.receive_actions(actions);
        bool doRender = true;

        if (gotPacket)
        {
            if (!networkActive) {
                // Python is driving now: the render is no longer the clock, so drop vsync
                networkActive = true;
                glfwSwapInterval(0);
            }

            // Handshake [-100.0, -100.0, difficulty?]: 25-Body Training Mode (train.py)
            if (actions.size() >= 2 && actions[0] == -100.0f && actions[1] == -100.0f) {
                activeBodies = MAX_BODIES;
                autoDisturb = true;
                difficulty = (actions.size() >= 3 && std::isfinite(actions[2])) ? std::min(1.0f, std::max(0.0f, actions[2])) : 1.0f;
                glfwSetWindowTitle(engine.window, "Standing Ragdoll AI - 25-Body Training Mode (Random Shoves Active)");
                for (int i = 0; i < MAX_BODIES; i++) {
                    spawnX[i] = 120.0f + (i % 5) * 140.0f + ((i / 5) - 2) * 8.0f;
                    skeletons[i].init(spawnX[i], true, (rand() % 2 == 0));
                    disturbers[i].active = false;
                }
                net.send_state(packState(activeBodies, false));
            }
            // Handshake [-101.0, -101.0, stress?]: 1-Body Solo Test Mode (test.py)
            // An optional third float > 0 turns on automatic shoves + virtual mouse balls at that difficulty.
            else if (actions.size() >= 2 && actions[0] == -101.0f && actions[1] == -101.0f) {
                activeBodies = 1;
                float stress = (actions.size() >= 3 && std::isfinite(actions[2])) ? actions[2] : 0.0f;
                autoDisturb = stress > 0.0f;
                difficulty = std::min(1.0f, std::max(0.0f, stress));
                glfwSetWindowTitle(engine.window, "Standing Ragdoll AI - Solo Test Mode (Press 'R' for Random Pose | Cursor Ball | Spacebar)");
                spawnX[0] = 400.0f;
                skeletons[0].init(spawnX[0], true, true);
                disturbers[0].active = false;
                net.send_state(packState(activeBodies, false));
            }
            // Step action packet: exactly 10 floats per active body, optionally +1 trailing difficulty float
            else if (actions.size() == (size_t)(activeBodies * 10) || actions.size() == (size_t)(activeBodies * 10 + 1)) {
                if (actions.size() == (size_t)(activeBodies * 10 + 1) && activeBodies > 1 && std::isfinite(actions.back()))
                    difficulty = std::min(1.0f, std::max(0.0f, actions.back()));

                for (int i = 0; i < activeBodies; i++) {
                    std::vector<float> bodyActions(actions.begin() + i * 10, actions.begin() + (i + 1) * 10);
                    skeletons[i].setActions(bodyActions);
                }

                // Track mouse velocity
                mouseVel = (mousePos - prevMousePos) * 60.0f;
                prevMousePos = mousePos;

                if (autoDisturb) {
                    for (int s = 0; s < activeBodies; s++) spawnDisturbances(s);
                }

                // Step physics in lockstep with Python: exactly 1 step (1/60s) per network command!
                float dt = 1.0f / 60.0f;
                const int subSteps = 4;
                float subDt = dt / (float)subSteps;
                for (int step = 0; step < subSteps; step++) {
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

                // Check resets and send state to Python
                net.send_state(packState(activeBodies, true));
                lastTime = glfwGetTime();

                // Training runs far faster than real time: only draw every 4th step
                stepCounter++;
                if (activeBodies > 1 && (stepCounter % 4) != 0) doRender = false;
            }
        }
        else if (networkActive)
        {
            // Python is driving: sleep until its next command instead of spinning, but redraw
            // every few ms of idleness so the window stays responsive.
            if (net.wait_readable(5)) {
                glfwPollEvents();
                continue;
            }
        }
        // Standalone freeplay mode: if Python is not running, run real-time physics for mouse testing
        else
        {
            double currentTime = glfwGetTime();
            float dt = (float)(currentTime - lastTime);
            lastTime = currentTime;
            if (dt > 0.033f) dt = 0.033f;

            mouseVel = (mousePos - prevMousePos) * 60.0f;
            prevMousePos = mousePos;

            const int subSteps = 4;
            float subDt = dt / (float)subSteps;
            for (int step = 0; step < subSteps; step++) {
                for (int s = 0; s < activeBodies; s++) {
                    int dragIdx = (s == draggedSkeletonIndex) ? draggedBoneIndex : -1;
                    skeletons[s].updatePhysics(subDt, dragIdx, mousePos);
                }

                if (activeBodies == 1) {
                    interactCursorBall(skeletons[0], mousePos, mouseVel, CURSOR_BALL_RADIUS);
                }
            }
        }

        // 3. Rendering
        if (doRender)
        {
            int fbW = 0, fbH = 0;
            glfwGetFramebufferSize(engine.window, &fbW, &fbH);
            glViewport(0, 0, fbW, fbH);

            glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // Ground line (floor at y = 20 px)
            glColor3f(0.35f, 0.35f, 0.40f);
            glLineWidth(2.0f);
            glBegin(GL_LINES);
            glVertex2f(0.0f, 20.0f);
            glVertex2f(800.0f, 20.0f);
            glEnd();

            if (activeBodies == 1) {
                // SOLO TEST MODE: 1 centered ragdoll
                // Target standing height guideline at y = 160 px (dashed guideline)
                glColor4f(0.20f, 0.80f, 0.50f, 0.35f);
                glBegin(GL_LINES);
                glVertex2f(280.0f, 160.0f);
                glVertex2f(520.0f, 160.0f);
                glEnd();

                // Ground center notch (x = 400, y = 20)
                glColor4f(0.40f, 0.60f, 0.80f, 0.50f);
                glBegin(GL_LINES);
                glVertex2f(400.0f, 12.0f);
                glVertex2f(400.0f, 28.0f);
                glEnd();

                // Render single centered ragdoll in high-contrast cyan/teal
                skeletons[0].draw(0.15f, 0.85f, 0.95f, 1.0f);

                // Stress-test ball (automatic pushes)
                if (autoDisturb && disturbers[0].active) {
                    glColor4f(0.95f, 0.25f, 0.25f, 0.85f);
                    drawDisc(disturbers[0].pos, CURSOR_BALL_RADIUS, 24);
                }
            } else {
                // 25-BODY TRAINING MODE
                for (int i = 0; i < activeBodies; i++) {
                    if (i == 0) {
                        skeletons[i].draw(1.0f, 0.90f, 0.20f, 1.0f); // Leader (bright gold)
                    } else {
                        float hue = (float)i / (float)activeBodies;
                        float r = 0.5f + 0.5f * cosf(hue * 6.28f);
                        float g = 0.5f + 0.5f * cosf(hue * 6.28f + 2.09f);
                        float b = 0.5f + 0.5f * cosf(hue * 6.28f + 4.18f);
                        skeletons[i].draw(r, g, b, 0.45f); // Distinct colors for each individual
                    }
                }
            }

            // 4. Draw glowing interactive cursor ball
            glColor4f(1.0f, 0.45f, 0.15f, 0.85f); // glowing bright orange / neon
            drawDisc(mousePos, CURSOR_BALL_RADIUS, 24);

            glColor4f(1.0f, 0.90f, 0.30f, 0.95f); // bright golden rim
            glLineWidth(2.0f);
            glBegin(GL_LINE_LOOP);
            for (int seg = 0; seg <= 24; seg++) {
                float th = 2.0f * M_PI * (float)seg / 24.0f;
                glVertex2f(mousePos.x + CURSOR_BALL_RADIUS * cosf(th), mousePos.y + CURSOR_BALL_RADIUS * sinf(th));
            }
            glEnd();

            glfwSwapBuffers(engine.window);
        }
        glfwPollEvents();
    }

    engine.shutdown();
    return 0;
}
