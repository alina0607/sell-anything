#include "sa/Engine.h"

#include <chrono>
#include <utility>

namespace sa
{

Engine::Engine(EngineConfig config) : config_(config) {}

void Engine::SetScene(std::unique_ptr<Scene> scene)
{
    if (scene_)
        scene_->OnExit();
    scene_ = std::move(scene);
    if (scene_)
        scene_->OnEnter();
}

void Engine::Run()
{
    using Clock = std::chrono::steady_clock;

    running_ = true;
    auto previous = Clock::now();
    double accumulator = 0.0;

    // Fixed-timestep update, variable-rate render.
    while (running_ && scene_)
    {
        const auto now = Clock::now();
        accumulator += std::chrono::duration<double>(now - previous).count();
        previous = now;

        while (accumulator >= config_.fixedTimestep && running_)
        {
            scene_->Update(config_.fixedTimestep);
            accumulator -= config_.fixedTimestep;
        }

        if (running_)
            scene_->Render();
    }

    if (scene_)
        scene_->OnExit();
}

} // namespace sa
