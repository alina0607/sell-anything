#pragma once

#include "sa/Backend.h"

#include <memory>

namespace sa
{

class Engine;

// A unit of gameplay (main menu, drawing canvas, buyer dialogue, ...).
class Scene
{
public:
    virtual ~Scene() = default;

    virtual void OnEnter(Engine&) {}
    virtual void OnExit(Engine&) {}
    virtual void Update(Engine& engine, double dt) = 0;
    virtual void Render(Engine& engine) = 0;
};

struct EngineConfig
{
    WindowDesc window;
    Color clearColor{0.96f, 0.95f, 0.92f, 1.0f};
    // Longest frame the scene will see, so a breakpoint or window drag doesn't explode dt.
    double maxDeltaTime = 0.1;
};

// Owns the backend, the main loop and the active scene.
class Engine
{
public:
    Engine(std::unique_ptr<Backend> backend, EngineConfig config = {});
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void SetScene(std::unique_ptr<Scene> scene);
    // Returns false if the backend failed to start.
    bool Run();
    void Quit() { running_ = false; }

    Backend& GetBackend() { return *backend_; }

private:
    std::unique_ptr<Backend> backend_;
    EngineConfig config_;
    std::unique_ptr<Scene> scene_;
    std::unique_ptr<Scene> pendingScene_;
    bool running_ = false;
};

} // namespace sa
