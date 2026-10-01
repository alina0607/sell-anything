#pragma once

#include "sa/Backend.h"

#include <string>

struct DGL_Mesh;

namespace sa
{

// DigiPen Graphics Library (Direct3D 11, Windows only).
// DGL is not redistributable, so it is never committed; see docs/backends.md.
class DGLBackend final : public Backend
{
public:
    static constexpr std::string_view kName = "dgl";

    std::string_view Name() const override { return kName; }

    bool Init(const WindowDesc& desc) override;
    void Shutdown() override;

    bool BeginFrame(Color clear) override;
    void EndFrame() override;

    Vec2 MousePosition() const override;
    bool MouseDown(MouseButton button) const override;
    bool KeyPressed(Key key) const override;

    void DrawLine(Vec2 from, Vec2 to, float thickness, Color color) override;
    void DrawRect(Vec2 topLeft, Vec2 size, Color color) override;

private:
    void DrawQuad(Vec2 worldCenter, Vec2 worldSize, float radians, Color color);

    std::string title_; // DGL keeps the pointer, so it must outlive the window.
    DGL_Mesh* unitQuad_ = nullptr;
    bool initialized_ = false;
};

} // namespace sa
