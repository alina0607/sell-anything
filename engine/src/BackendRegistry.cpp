#include "sa/Backend.h"

#include "backends/NullBackend.h"
#if SA_HAS_DGL
#include "backends/DGLBackend.h"
#endif
#if SA_HAS_OPENGL
#include "backends/OpenGLBackend.h"
#endif

namespace sa
{

std::vector<std::string_view> AvailableBackends()
{
    std::vector<std::string_view> names;
#if SA_HAS_DGL
    names.push_back(DGLBackend::kName);
#endif
#if SA_HAS_OPENGL
    names.push_back(OpenGLBackend::kName);
#endif
    names.push_back(NullBackend::kName);
    return names;
}

std::unique_ptr<Backend> CreateBackend(std::string_view name)
{
    if (name.empty())
        name = AvailableBackends().front();

#if SA_HAS_DGL
    if (name == DGLBackend::kName)
        return std::make_unique<DGLBackend>();
#endif
#if SA_HAS_OPENGL
    if (name == OpenGLBackend::kName)
        return std::make_unique<OpenGLBackend>();
#endif
    if (name == NullBackend::kName)
        return std::make_unique<NullBackend>();
    return nullptr;
}

} // namespace sa
