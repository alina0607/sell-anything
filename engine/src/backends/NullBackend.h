#pragma once

#include "sa/Backend.h"

namespace sa
{

// Headless backend: no window, draws nothing, closes itself after a fixed number of frames.
// Used for CI smoke tests and for running the game logic on machines without a GPU.
class NullBackend final : public Backend
{
public:
    static constexpr std::string_view kName = "null";

    explicit NullBackend(int maxFrames = 120) : maxFrames_(maxFrames) {}

    std::string_view Name() const override { return kName; }

    bool Init(const WindowDesc&) override { return true; }
    void Shutdown() override {}

    bool BeginFrame(Color) override { return frame_++ < maxFrames_; }
    void EndFrame() override {}

    Vec2 MousePosition() const override { return {}; }
    bool MouseDown(MouseButton) const override { return false; }
    bool KeyPressed(Key) const override { return false; }

    void DrawLine(Vec2, Vec2, float, Color) override {}
    void DrawRect(Vec2, Vec2, Color) override {}

private:
    int maxFrames_;
    int frame_ = 0;
};

} // namespace sa
