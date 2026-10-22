#include <ctime>
#include <cstdlib>
#include <vector>

#include "platform/Engine.h"
#include "platform/Network.h"
#include "World.h"
#include "Input.h"
#include "Protocol.h"
#include "Render.h"
#include "Simulation.h"

int main(void)
{
    srand((unsigned)time(NULL));

    Engine engine(800, 600, "Standing Ragdoll AI");
    if (!engine.init())
        return -1;

    // Sockets: C++ listens on STANDUP_SIM_PORT (5005), sends to Python on STANDUP_PY_PORT (5006).
    // Override both (and the matching vars for train.py/test.py) to run a second instance alongside.
    NetworkData net("127.0.0.1", envInt("STANDUP_SIM_PORT", 5005), envInt("STANDUP_PY_PORT", 5006));

    registerInputCallbacks(engine.window);

    // Initial setup: Body 0 is centered at x = 400.0f with random limbs enabled
    spawnX[0] = 400.0f;
    skeletons[0].init(spawnX[0], true, true);
    for (int i = 1; i < MAX_BODIES; i++) {
        spawnX[i] = trainingSpawnX(i);
        skeletons[i].init(spawnX[i], true, (rand() % 2 == 0));
    }

    setupRenderState();

    double lastTime = glfwGetTime();
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

            if (handleHandshake(engine.window, net, actions)) {
                // mode switched, initial state already sent
            }
            else if (isStepPacket(actions)) {
                applyStepPacket(actions);

                // Step physics in lockstep with Python: exactly 1 step (1/60s) per network command!
                stepLockstep();

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
        else
        {
            // Standalone freeplay mode: if Python is not running, run real-time physics for mouse testing
            double currentTime = glfwGetTime();
            stepFreeplay((float)(currentTime - lastTime));
            lastTime = currentTime;
        }

        // 2. Rendering
        if (doRender)
            renderFrame(engine.window);
        glfwPollEvents();
    }

    engine.shutdown();
    return 0;
}
