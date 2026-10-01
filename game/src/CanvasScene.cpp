#include "CanvasScene.h"

#include <cmath>

namespace
{

constexpr float kBrushThickness = 6.0f;
// Skip points closer than this to the previous one; keeps strokes small without losing shape.
constexpr float kMinPointSpacing = 2.0f;
constexpr sa::Color kInk{0.12f, 0.12f, 0.14f, 1.0f};

float Distance(sa::Vec2 a, sa::Vec2 b)
{
    const sa::Vec2 d = b - a;
    return std::sqrt(d.x * d.x + d.y * d.y);
}

} // namespace

void CanvasScene::Update(sa::Engine& engine, double /*dt*/)
{
    sa::Backend& backend = engine.GetBackend();

    if (backend.KeyPressed(sa::Key::Escape))
    {
        engine.Quit();
        return;
    }
    if (backend.KeyPressed(sa::Key::C) || backend.MouseDown(sa::MouseButton::Right))
    {
        strokes_.clear();
        drawing_ = false;
        return;
    }

    const bool down = backend.MouseDown(sa::MouseButton::Left);
    const sa::Vec2 mouse = backend.MousePosition();

    if (down && !drawing_)
        strokes_.push_back({mouse});
    else if (down && Distance(strokes_.back().back(), mouse) >= kMinPointSpacing)
        strokes_.back().push_back(mouse);
    drawing_ = down;
}

void CanvasScene::Render(sa::Engine& engine)
{
    sa::Backend& backend = engine.GetBackend();

    for (const Stroke& stroke : strokes_)
    {
        if (stroke.size() == 1)
            backend.DrawLine(stroke[0], stroke[0], kBrushThickness, kInk);
        for (size_t i = 1; i < stroke.size(); ++i)
            backend.DrawLine(stroke[i - 1], stroke[i], kBrushThickness, kInk);
    }
}
