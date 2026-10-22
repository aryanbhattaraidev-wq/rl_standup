#pragma once

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <string>
#include <iostream>

struct Engine
{
    GLFWwindow* window = nullptr;
    int width, height;
    std::string title;

    Engine(int w, int h, const std::string& t) : width(w), height(h), title(t) {}

    bool init()
    {
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return false;
        }
        window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
        if (!window) {
            std::cerr << "Failed to create window" << std::endl;
            glfwTerminate();
            return false;
        }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1); // vsync while idle; rl_standup.cpp turns it off once Python is driving
        glViewport(0, 0, width, height);
        return true;
    }

    void shutdown()
    {
        glfwTerminate();
        window = nullptr;
    }
};
