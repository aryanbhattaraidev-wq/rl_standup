#pragma once

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct vec2
{
    float x, y;

    vec2() : x(0), y(0) {}
    vec2(float x, float y) : x(x), y(y) {}
    vec2(float v) : x(v), y(v) {}

    vec2 operator+(const vec2& o) const { return vec2(x + o.x, y + o.y); }
    vec2 operator-(const vec2& o) const { return vec2(x - o.x, y - o.y); }
    vec2 operator*(float s) const { return vec2(x * s, y * s); }
    vec2 operator/(float s) const { return vec2(x / s, y / s); }
};

inline vec2 rotate(const vec2& v, float angle)
{
    float c = cosf(angle), s = sinf(angle);
    return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}
