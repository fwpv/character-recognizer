#pragma once

#include <string>
#include <string_view>
#include <variant>
#include <vector>

struct TrainCommand {
    std::string db_path;
    std::string path_to_save;
};

struct RecognizeCommand {
    std::string target_path;
    std::string snn_data_path;
};

struct HelpCommand {
};

using Command = std::variant<std::monostate,
    TrainCommand, RecognizeCommand, HelpCommand>;

Command ParseStrings(const std::vector<std::string_view>& strings);
void InterpretCommand(Command command);
void ProcessInput(int argc, char** argv);