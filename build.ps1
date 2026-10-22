# Builds rl_standup.exe. Needs a MinGW-w64 g++ on PATH (for example WinLibs).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

g++ -O2 -std=c++17 sim/main.cpp -o rl_standup.exe `
    -I sim -I third_party/GLFW/include `
    third_party/GLFW/lib/libglfw3.a -lopengl32 -lgdi32 -lws2_32

Write-Host "Built rl_standup.exe"
