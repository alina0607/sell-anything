#include "DGLBackend.h"

#include <DGL.h>

#include <cmath>
#include <iostream>

namespace sa
{

namespace
{

LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    int result = 0;
    if (DGL_System_HandleWindowsMessage(message, wParam, lParam, &result))
        return result;
    return DefWindowProc(window, message, wParam, lParam);
}

int ToVirtualKey(Key key)
{
    switch (key)
    {
    case Key::Escape: return VK_ESCAPE;
    case Key::Enter: return VK_RETURN;
    case Key::Space: return VK_SPACE;
    case Key::Backspace: return VK_BACK;
    case Key::C: return 'C';
    }
    return 0;
}

// DGL draws in world space: origin at the window center, y up, 1 unit = 1 pixel at zoom 1.
Vec2 PixelToWorld(Vec2 pixel)
{
    const DGL_Vec2 screen{pixel.x, pixel.y};
    const DGL_Vec2 world = DGL_Camera_ScreenCoordToWorld(&screen);
    return {world.x, world.y};
}

} // namespace

bool DGLBackend::Init(const WindowDesc& desc)
{
    title_ = desc.title;

    DGL_SysInitInfo info{};
    info.mAppInstance = GetModuleHandle(nullptr);
    info.mShow = SW_SHOW;
    info.mWindowWidth = static_cast<unsigned>(desc.width);
    info.mWindowHeight = static_cast<unsigned>(desc.height);
    info.mMaxFrameRate = 60;
    info.mClassStyle = CS_HREDRAW | CS_VREDRAW;
    info.mWindowStyle = WS_OVERLAPPEDWINDOW;
    info.mWindowTitle = title_.c_str();
    info.mCreateConsole = FALSE;
    info.mWindowIcon = 0;
    info.pWindowsCallback = WindowProc;

    if (!DGL_System_Init(&info))
    {
        std::cerr << "DGL_System_Init failed: " << DGL_System_GetLastError() << '\n';
        return false;
    }
    initialized_ = true;

    // One white unit quad, scaled/rotated/tinted per draw, so nothing is allocated per frame.
    const DGL_Color white{1.0f, 1.0f, 1.0f, 1.0f};
    const DGL_Vec2 uv{0.0f, 0.0f};
    const DGL_Vec2 p0{-0.5f, -0.5f}, p1{0.5f, -0.5f}, p2{0.5f, 0.5f}, p3{-0.5f, 0.5f};
    DGL_Graphics_StartMesh();
    DGL_Graphics_AddTriangle(&p0, &white, &uv, &p1, &white, &uv, &p2, &white, &uv);
    DGL_Graphics_AddTriangle(&p0, &white, &uv, &p2, &white, &uv, &p3, &white, &uv);
    unitQuad_ = DGL_Graphics_EndMesh();

    DGL_Graphics_SetBlendMode(DGL_BM_BLEND);
    return unitQuad_ != nullptr;
}

void DGLBackend::Shutdown()
{
    if (!initialized_)
        return;
    DGL_Graphics_FreeMesh(&unitQuad_);
    DGL_System_Exit();
    initialized_ = false;
}

bool DGLBackend::BeginFrame(Color clear)
{
    DGL_System_FrameControl();
    DGL_System_Update();
    if (!DGL_System_DoesWindowExist())
        return false;

    const DGL_Color background{clear.r, clear.g, clear.b, clear.a};
    DGL_Graphics_SetBackgroundColor(&background);
    DGL_Graphics_StartDrawing();
    DGL_Graphics_SetShaderMode(DGL_PSM_COLOR, DGL_VSM_DEFAULT);
    return true;
}

void DGLBackend::EndFrame() { DGL_Graphics_FinishDrawing(); }

Vec2 DGLBackend::MousePosition() const
{
    const DGL_Vec2 p = DGL_Input_GetMousePosition();
    return {p.x, p.y};
}

bool DGLBackend::MouseDown(MouseButton button) const
{
    return DGL_Input_KeyDown(button == MouseButton::Left ? VK_LBUTTON : VK_RBUTTON);
}

bool DGLBackend::KeyPressed(Key key) const
{
    return DGL_Input_KeyTriggered(static_cast<unsigned char>(ToVirtualKey(key)));
}

void DGLBackend::DrawLine(Vec2 from, Vec2 to, float thickness, Color color)
{
    const Vec2 a = PixelToWorld(from);
    const Vec2 b = PixelToWorld(to);
    const Vec2 d = b - a;
    const float length = std::sqrt(d.x * d.x + d.y * d.y);
    // Extend by the thickness so consecutive segments overlap instead of leaving gaps.
    DrawQuad((a + b) * 0.5f, {length + thickness, thickness}, std::atan2(d.y, d.x), color);
}

void DGLBackend::DrawRect(Vec2 topLeft, Vec2 size, Color color)
{
    const Vec2 a = PixelToWorld(topLeft);
    const Vec2 b = PixelToWorld(topLeft + size);
    DrawQuad((a + b) * 0.5f, {std::fabs(b.x - a.x), std::fabs(b.y - a.y)}, 0.0f, color);
}

void DGLBackend::DrawQuad(Vec2 worldCenter, Vec2 worldSize, float radians, Color color)
{
    const DGL_Vec2 position{worldCenter.x, worldCenter.y};
    const DGL_Vec2 scale{worldSize.x, worldSize.y};
    // A tint with alpha 1 fully replaces the white mesh color.
    const DGL_Color tint{color.r, color.g, color.b, 1.0f};
    DGL_Graphics_SetCB_TransformData(&position, &scale, radians);
    DGL_Graphics_SetCB_TintColor(&tint);
    DGL_Graphics_SetCB_Alpha(color.a);
    DGL_Graphics_DrawMesh(unitQuad_, DGL_DM_TRIANGLELIST);
}

} // namespace sa
