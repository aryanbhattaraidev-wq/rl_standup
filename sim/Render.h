#pragma once

#include <cmath>
#include "platform/Engine.h"
#include "World.h"
#include "Disturbances.h"

inline void drawDisc(vec2 c, float r, int segs)
{
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(c.x, c.y);
    for (int seg = 0; seg <= segs; seg++) {
        float th = 2.0f * M_PI * (float)seg / (float)segs;
        glVertex2f(c.x + r * cosf(th), c.y + r * sinf(th));
    }
    glEnd();
}

// Fixed 800x600 world, bottom-left origin, drawn with alpha blending
inline void setupRenderState()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WORLD_W, 0, WORLD_H, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

inline void renderFrame(GLFWwindow* window)
{
    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    glViewport(0, 0, fbW, fbH);

    glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Ground line (floor at y = 20 px)
    glColor3f(0.35f, 0.35f, 0.40f);
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glVertex2f(0.0f, 20.0f);
    glVertex2f(WORLD_W, 20.0f);
    glEnd();

    if (activeBodies == 1) {
        // SOLO TEST MODE: 1 centered ragdoll
        // Target standing height guideline at y = 160 px
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

    // Glowing interactive cursor ball
    glColor4f(1.0f, 0.45f, 0.15f, 0.85f);
    drawDisc(mousePos, CURSOR_BALL_RADIUS, 24);

    glColor4f(1.0f, 0.90f, 0.30f, 0.95f); // bright golden rim
    glLineWidth(2.0f);
    glBegin(GL_LINE_LOOP);
    for (int seg = 0; seg <= 24; seg++) {
        float th = 2.0f * M_PI * (float)seg / 24.0f;
        glVertex2f(mousePos.x + CURSOR_BALL_RADIUS * cosf(th), mousePos.y + CURSOR_BALL_RADIUS * sinf(th));
    }
    glEnd();

    glfwSwapBuffers(window);
}
