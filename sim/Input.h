#pragma once

#include "platform/Engine.h"
#include "World.h"
#include "Disturbances.h"

// GLFW input callbacks: left-click drags a limb, Space shoves the solo ragdoll, R respawns it.

inline void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
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

inline void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
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

inline void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
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

inline void registerInputCallbacks(GLFWwindow* window)
{
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetKeyCallback(window, keyCallback);
}
