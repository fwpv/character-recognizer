#pragma once

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
    enum Algorithm {
        SEQUENTIALLY,
        SHUFFLED, // Not supported yet
        SHUFFLED_WITH_NOT_SYM // Not supported yet
    };
    std::string snn_data_path = ""s; // Not supported yet
    std::string db_path = "training_chars"s;
    std::string path_to_save = "snn_data"s;
    int training_cycles = 1000; // Not supported yet
    Algorithm algorithm = SEQUENTIALLY; // Not supported yet
};

struct RecognizeCommand {
    enum OutputType {
        TERMINAL,
        FILE // Not supported yet
    };
    std::string snn_data_path = "snn_data"s;
    std::string target_path = "target_chars"s; // Only the picture is supported so far
    std::string result_path = "result.txt"s; // Not supported yet
    OutputType output_type = TERMINAL; // Not supported yet
};

struct HelpCommand {
};

using Command = std::variant<std::monostate,
    TrainCommand, RecognizeCommand, HelpCommand>;

Command ParseStrings(const std::vector<std::string_view>& strings);
void InterpretCommand(Command command);
void ProcessInput(int argc, char** argv);