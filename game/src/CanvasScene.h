#pragma once

#include "sa/Engine.h"

#include <vector>

// Free drawing with the mouse. Strokes are kept in window pixels, ready to send to the
// sketch classifier (see docs/protocol.md).
class CanvasScene final : public sa::Scene
{
public:
    using Stroke = std::vector<sa::Vec2>;

    void Update(sa::Engine& engine, double dt) override;
    void Render(sa::Engine& engine) override;

    const std::vector<Stroke>& Strokes() const { return strokes_; }

private:
    std::vector<Stroke> strokes_;
    bool drawing_ = false;
};
