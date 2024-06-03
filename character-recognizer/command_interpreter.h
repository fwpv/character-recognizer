#pragma once

#include "request_handler.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

class ParsingError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

using namespace std::literals;

struct TrainCommand {
    std::string snn_data_path = ""s;
    std::string db_path = "training_chars"s;
    std::string path_to_save = "snn_data"s;
    int training_cycles = 1000;
    RequestHandler::Algorithm algorithm = RequestHandler::SHUFFLED;
    int hidden_neurons = 32;
};

struct RecognizeCommand {
    std::string snn_data_path = "snn_data"s;
    std::string target_path = "target_chars"s;
    std::string result_path = ""s;
};

struct HelpCommand {
};

using Command = std::variant<std::monostate,
    TrainCommand, RecognizeCommand, HelpCommand>;

Command ParseStrings(const std::vector<std::string_view>& strings);
void InterpretCommand(Command command);
void ProcessInput(int argc, char** argv);