#include "CanvasScene.h"

#include "sa/Engine.h"

#include <iostream>
#include <memory>
#include <string_view>
#include <utility>

namespace
{

void PrintUsage()
{
    std::cout << "Usage: sell_anything [--backend NAME] [--list-backends]\n"
                 "Backends in this build:";
    for (std::string_view name : sa::AvailableBackends())
        std::cout << ' ' << name;
    std::cout << " (default: " << sa::AvailableBackends().front() << ")\n";
}

} // namespace

int main(int argc, char** argv)
{
    std::string_view backendName;

    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if (arg == "--backend" && i + 1 < argc)
            backendName = argv[++i];
        else if (arg == "--list-backends" || arg == "--help")
        {
            PrintUsage();
            return 0;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << '\n';
            PrintUsage();
            return 2;
        }
    }

    auto backend = sa::CreateBackend(backendName);
    if (!backend)
    {
        std::cerr << "Backend '" << backendName << "' is not available in this build\n";
        PrintUsage();
        return 2;
    }
    std::cout << "Using backend: " << backend->Name() << '\n';

    sa::Engine engine(std::move(backend));
    engine.SetScene(std::make_unique<CanvasScene>());
    return engine.Run() ? 0 : 1;
}
