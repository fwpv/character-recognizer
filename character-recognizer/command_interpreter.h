#pragma once

#include <string_view>
#include <variant>
#include <vector>

struct TrainCommand {
};

struct RecognizeCommand {
};

struct HelpCommand {
};

using Command = std::variant<std::monostate,
    TrainCommand, RecognizeCommand, HelpCommand>;

//Command ExtractСommand(const std::vector<std::string_view>& strings);
//void InterpretCommand(Command command);
void ProcessInput(int argc, char** argv);