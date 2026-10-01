#include "sa/Engine.h"

#include <iostream>
#include <memory>

namespace
{

// Placeholder until the window and renderer exist: runs for one second, then quits.
class BootScene final : public sa::Scene
{
public:
    explicit BootScene(sa::Engine& engine) : engine_(engine) {}

    void OnEnter() override { std::cout << "Sell Anything — engine booted\n"; }

    void Update(double dt) override
    {
        elapsed_ += dt;
        if (elapsed_ >= 1.0)
            engine_.Quit();
    }

    void Render() override {}

private:
    sa::Engine& engine_;
    double elapsed_ = 0.0;
};

} // namespace

int main()
{
    sa::Engine engine;
    engine.SetScene(std::make_unique<BootScene>(engine));
    engine.Run();
    return 0;
}
