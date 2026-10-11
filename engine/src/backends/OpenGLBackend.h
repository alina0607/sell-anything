#pragma once

#include "sa/Backend.h"

#include <array>
#include <string>
#include <vector>

struct GLFWwindow;

namespace sa
{

// GLFW window + OpenGL 3.3 core. Runs on Windows, macOS and Linux.
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
    bool KeyDown(Key key) const override;

    void DrawLine(Vec2 from, Vec2 to, float thickness, Color color) override;
    void DrawRect(Vec2 topLeft, Vec2 size, Color color) override;

    TextureId LoadTexture(const std::string& path, TextureFilter filter) override;
    Vec2 TextureSize(TextureId texture) const override;
    void DrawSprite(TextureId texture, UVRect source, Vec2 topLeft, Vec2 size, float alpha) override;
    void DrawSpriteFilled(TextureId texture, UVRect source, Vec2 topLeft, Vec2 size, Color color) override;

private:
    struct Vertex
    {
        float x, y;
        float u, v;
        float r, g, b, a;
    };
    struct Texture
    {
        unsigned gl = 0;
        Vec2 size;
    };
    // Consecutive quads with the same texture, drawn with one call.
    struct Batch
    {
        unsigned texture = 0;
        size_t first = 0;
        size_t count = 0;
    };

    static constexpr int kKeyCount = static_cast<int>(Key::Down) + 1;

    void PushQuad(const std::array<Vec2, 4>& corners, Color color, unsigned texture, UVRect uv);
    static unsigned CreateTexture(const unsigned char* rgba, int width, int height,
                                  TextureFilter filter = TextureFilter::Pixel);

    GLFWwindow* window_ = nullptr;
    unsigned program_ = 0;
    unsigned vao_ = 0;
    unsigned vbo_ = 0;
    int screenSizeLocation_ = -1;
    std::vector<Vertex> vertices_;
    std::vector<Batch> batches_;
    std::vector<Texture> textures_;   // TextureId n is textures_[n - 1]
    unsigned whiteTexture_ = 0;
    std::array<bool, kKeyCount> keyDown_{};
    std::array<bool, kKeyCount> keyPressed_{};
    std::string typed_;         // characters typed during the last frame
    std::string typing_;        // characters arriving for the next frame

    // Testing hooks (see docs/backends.md): captures, bounded runs, and deterministic held keys.
    struct Capture
    {
        std::string path;
        unsigned frame = 0;
    };
    std::vector<Capture> captures_;
    struct HeldKey
    {
        int key = 0;
        unsigned first = 0;
        unsigned last = 0;
    };
    std::vector<HeldKey> heldKeys_;
    unsigned exitAfter_ = 0;
    unsigned frame_ = 0;

public:
    std::string TypedText() const override { return typed_; }
    void OnChar(unsigned codepoint);
};

} // namespace sa
