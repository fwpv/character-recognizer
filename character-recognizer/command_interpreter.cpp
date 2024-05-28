#include "command_interpreter.h"

#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace std::literals;

void ProcessInput(int argc, char** argv) {
    assert(argc > 0 && argv != nullptr);

    if (argc == 1) {
        assert(argv[0] != nullptr);
        std::filesystem::path path(argv[0]);
        std::string filename = path.filename().string();
        std::cout << "Type \'"s + filename + " help\' to get information!"s << std::endl;
        return;
    }

    std::vector<std::string_view> strings;
    for (int i = 1; i < argc; ++i) {
        assert(argv[i] != nullptr);
        strings.emplace_back(argv[i], std::strlen(argv[i]));
    }

    std::cout << "Entered: "s << std::endl;
    for (const auto& item : strings) {
        std::cout << item << std::endl;
    }
}