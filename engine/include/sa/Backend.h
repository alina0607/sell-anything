#pragma once

#include "sa/Math.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sa
{

enum class MouseButton
{
    Left,
    Right,
};

enum class Key
{
    Escape,
    Enter,
    Space,
    Backspace,
    C,
};

struct WindowDesc
{
    std::string title = "Sell Anything";
    int width = 1280;
    int height = 720;
};

// Everything the engine needs from the platform: a window, input, and 2D drawing.
// Coordinates are window pixels with the origin at the top-left and y pointing down,
// the same convention as the Quick, Draw! stroke data.
class Backend
{
public:
    virtual ~Backend() = default;

    virtual std::string_view Name() const = 0;

    virtual bool Init(const WindowDesc& desc) = 0;
    virtual void Shutdown() = 0;

    // Pumps window events and starts a frame. Returns false once the window is closed.
    virtual bool BeginFrame(Color clear) = 0;
    virtual void EndFrame() = 0;

    virtual Vec2 MousePosition() const = 0;
    virtual bool MouseDown(MouseButton button) const = 0;
    // True only on the frame the key went down.
    virtual bool KeyPressed(Key key) const = 0;

    virtual void DrawLine(Vec2 from, Vec2 to, float thickness, Color color) = 0;
    virtual void DrawRect(Vec2 topLeft, Vec2 size, Color color) = 0;
};

// Names of the backends compiled into this build, preferred one first.
std::vector<std::string_view> AvailableBackends();

// Creates a backend by name; an empty name picks the preferred one.
// Returns nullptr if the name is not compiled into this build.
std::unique_ptr<Backend> CreateBackend(std::string_view name = {});

} // namespace sa
