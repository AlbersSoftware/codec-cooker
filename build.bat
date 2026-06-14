@echo off
cd /d %~dp0

echo Building ImGui project...

C:\ProgramData\mingw64\mingw64\bin\g++.exe -std=c++17 -static-libgcc -static-libstdc++ ^
-Iexternal/imgui -Iexternal/backends -Ilibs/SDL3/include ^
src/main.cpp src/dct.cpp src/quantization.cpp src/entropy.cpp src/loopFilter.cpp ^
external/imgui/imgui.cpp external/imgui/imgui_draw.cpp external/imgui/imgui_tables.cpp external/imgui/imgui_widgets.cpp ^
external/backends/imgui_impl_sdl3.cpp external/backends/imgui_impl_opengl3.cpp ^
-Llibs/SDL3/lib -lSDL3 -lopengl32 -lcomdlg32 -lgdi32 ^
-o imgui_app.exe

if %errorlevel% neq 0 (
    echo BUILD FAILED
    pause
    exit /b
)

echo BUILD SUCCESS
imgui_app.exe
