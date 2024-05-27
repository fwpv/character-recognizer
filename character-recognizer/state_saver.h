#include "snn.h"

#include <filesystem>

void SaveSnnState(const std::filesystem::path& file, const SnnMemento& state);

SnnMemento LoadSnnState(const std::filesystem::path& file);