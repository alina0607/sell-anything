#pragma once

#include "sa/Backend.h"

#include <array>
#include <vector>

struct GLFWwindow;

namespace sa
{

// GLFW window + OpenGL 3.3 core. Used on macOS and Linux.
// All draws in a frame are batched into one vertex buffer and submitted in EndFrame.
class OpenGLBackend final : public Backend
{
public:
    static constexpr std::string_view kName = "opengl";

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
    struct Vertex
    {
        float x, y;
        float r, g, b, a;
    };

    static constexpr int kKeyCount = 5;

    void PushQuad(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, Color color);

    GLFWwindow* window_ = nullptr;
    unsigned program_ = 0;
    unsigned vao_ = 0;
    unsigned vbo_ = 0;
    int screenSizeLocation_ = -1;
    std::vector<Vertex> vertices_;
    std::array<bool, kKeyCount> keyDown_{};
    std::array<bool, kKeyCount> keyPressed_{};
};

} // namespace sa
