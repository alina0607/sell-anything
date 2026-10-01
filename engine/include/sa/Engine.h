#pragma once

#include <memory>

namespace sa
{

// A unit of gameplay (main menu, drawing canvas, buyer dialogue, ...).
class Scene
{
public:
    virtual ~Scene() = default;

    virtual void OnEnter() {}
    virtual void OnExit() {}
    virtual void Update(double dt) = 0;
    virtual void Render() = 0;
};

struct EngineConfig
{
    int windowWidth = 1280;
    int windowHeight = 720;
    double fixedTimestep = 1.0 / 60.0;
};

// Owns the main loop and the active scene.
class Engine
{
public:
    explicit Engine(EngineConfig config = {});

    void SetScene(std::unique_ptr<Scene> scene);
    void Run();
    void Quit() { running_ = false; }

private:
    EngineConfig config_;
    std::unique_ptr<Scene> scene_;
    bool running_ = false;
};

} // namespace sa
