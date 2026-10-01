#include "OpenGLBackend.h"

#if defined(__APPLE__)
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#endif

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstddef>
#include <iostream>

namespace sa
{

namespace
{

constexpr const char* kVertexShader = R"(#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aColor;
uniform vec2 uScreenSize;
out vec4 vColor;
void main()
{
    // Pixels (top-left origin, y down) to normalized device coordinates.
    vec2 ndc = vec2(aPos.x / uScreenSize.x * 2.0 - 1.0, 1.0 - aPos.y / uScreenSize.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    vColor = aColor;
}
)";

constexpr const char* kFragmentShader = R"(#version 330 core
in vec4 vColor;
out vec4 fragColor;
void main() { fragColor = vColor; }
)";

constexpr int kGlfwKeys[] = {GLFW_KEY_ESCAPE, GLFW_KEY_ENTER, GLFW_KEY_SPACE, GLFW_KEY_BACKSPACE,
                             GLFW_KEY_C};
static_assert(sizeof(kGlfwKeys) / sizeof(kGlfwKeys[0]) == static_cast<int>(Key::C) + 1,
              "kGlfwKeys must list every sa::Key in enum order");

unsigned CompileShader(GLenum type, const char* source)
{
    const unsigned shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "Shader compile error: " << log << '\n';
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

unsigned LinkProgram()
{
    const unsigned vs = CompileShader(GL_VERTEX_SHADER, kVertexShader);
    const unsigned fs = CompileShader(GL_FRAGMENT_SHADER, kFragmentShader);
    if (!vs || !fs)
        return 0;

    const unsigned program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::cerr << "Shader link error: " << log << '\n';
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

} // namespace

bool OpenGLBackend::Init(const WindowDesc& desc)
{
    if (!glfwInit())
    {
        std::cerr << "glfwInit failed\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE); // required on macOS

    window_ = glfwCreateWindow(desc.width, desc.height, desc.title.c_str(), nullptr, nullptr);
    if (!window_)
    {
        std::cerr << "glfwCreateWindow failed\n";
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    program_ = LinkProgram();
    if (!program_)
        return false;
    screenSizeLocation_ = glGetUniformLocation(program_, "uScreenSize");

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, x)));
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, r)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return true;
}

void OpenGLBackend::Shutdown()
{
    if (!window_)
        return;
    glDeleteBuffers(1, &vbo_);
    glDeleteVertexArrays(1, &vao_);
    glDeleteProgram(program_);
    glfwDestroyWindow(window_);
    glfwTerminate();
    window_ = nullptr;
}

bool OpenGLBackend::BeginFrame(Color clear)
{
    glfwPollEvents();
    if (glfwWindowShouldClose(window_))
        return false;

    for (int i = 0; i < kKeyCount; ++i)
    {
        const bool down = glfwGetKey(window_, kGlfwKeys[i]) == GLFW_PRESS;
        keyPressed_[i] = down && !keyDown_[i];
        keyDown_[i] = down;
    }

    // Framebuffer size differs from window size on HiDPI displays.
    int fbWidth = 0, fbHeight = 0;
    glfwGetFramebufferSize(window_, &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);
    glClearColor(clear.r, clear.g, clear.b, clear.a);
    glClear(GL_COLOR_BUFFER_BIT);

    vertices_.clear();
    return true;
}

void OpenGLBackend::EndFrame()
{
    if (!vertices_.empty())
    {
        int width = 0, height = 0;
        glfwGetWindowSize(window_, &width, &height);

        glUseProgram(program_);
        glUniform2f(screenSizeLocation_, static_cast<float>(width), static_cast<float>(height));
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices_.size() * sizeof(Vertex)),
                     vertices_.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices_.size()));
    }
    glfwSwapBuffers(window_);
}

Vec2 OpenGLBackend::MousePosition() const
{
    double x = 0.0, y = 0.0;
    glfwGetCursorPos(window_, &x, &y);
    return {static_cast<float>(x), static_cast<float>(y)};
}

bool OpenGLBackend::MouseDown(MouseButton button) const
{
    const int glfwButton =
        button == MouseButton::Left ? GLFW_MOUSE_BUTTON_LEFT : GLFW_MOUSE_BUTTON_RIGHT;
    return glfwGetMouseButton(window_, glfwButton) == GLFW_PRESS;
}

bool OpenGLBackend::KeyPressed(Key key) const { return keyPressed_[static_cast<int>(key)]; }

void OpenGLBackend::DrawLine(Vec2 from, Vec2 to, float thickness, Color color)
{
    Vec2 d = to - from;
    const float length = std::sqrt(d.x * d.x + d.y * d.y);
    if (length < 1e-4f)
        d = {1.0f, 0.0f}; // a click without movement still leaves a dot
    else
        d = d * (1.0f / length);

    // Extend by half the thickness at both ends so consecutive segments overlap.
    const float h = thickness * 0.5f;
    const Vec2 along = d * h;
    const Vec2 normal{-d.y * h, d.x * h};
    const Vec2 a = from - along;
    const Vec2 b = to + along;
    PushQuad(a + normal, b + normal, b - normal, a - normal, color);
}

void OpenGLBackend::DrawRect(Vec2 topLeft, Vec2 size, Color color)
{
    PushQuad(topLeft, {topLeft.x + size.x, topLeft.y}, topLeft + size,
             {topLeft.x, topLeft.y + size.y}, color);
}

void OpenGLBackend::PushQuad(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, Color c)
{
    for (const Vec2& p : {p0, p1, p2, p0, p2, p3})
        vertices_.push_back({p.x, p.y, c.r, c.g, c.b, c.a});
}

} // namespace sa
