#include "command_interpreter.h"

#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

std::pair<std::string_view, std::string_view>
        ParseParameter(std::string_view parameter) {
    size_t equal_pos = parameter.find_first_of('=');
    if (equal_pos == std::string_view::npos) {
        throw ParsingError("There is no '=' sign after parameter"s);
    }
    if (equal_pos == 0) {
        throw ParsingError("Empty string before '=' sign"s);
    }
    if (equal_pos == parameter.size() - 1) {
        throw ParsingError("Empty string after '=' sign"s);
    }

    std::string_view name = parameter.substr(0, equal_pos);

    // Remove minus
    if (name[0] != '-') {
        throw ParsingError("Parameter must start with '-'"s);
    }
    name.remove_prefix(1);
    if (name.empty()) {
        throw ParsingError("An empty parameter"s);
    }

    std::string_view value =
        parameter.substr(equal_pos + 1, parameter.size() - equal_pos - 1);

    // Remove quotes
    if (value.front() == '"') {
        value.remove_prefix(1);
    }
    if (!value.empty() && value.back() == '"') {
        value.remove_suffix(1);
    }
    if (value.empty()) {
        throw ParsingError("Empty string after '=' sign"s);
    }

    return {name, value};
}

int StringViewToInt(std::string_view value) {
    try {
        return std::stoi(std::string(value));
    } catch (...) {
        throw std::invalid_argument(std::string(value) + " is not an integer value"s);
    }
}

}

Command ParseStrings(const std::vector<std::string_view>& strings) {
    assert(!strings.empty());

    std::string_view name = strings[0];

    Command command;

    if (name == "help"sv) {
        command = HelpCommand();

    } else if (name == "train"sv) {
        TrainCommand train_command;
        for (int i = 1; i < strings.size(); ++i) {
            std::string_view str = strings[i];
            auto [name, value] = ParseParameter(str);

            if (name == "snn_data_path"sv) {
                train_command.snn_data_path = std::string(value);

            } else if (name == "db_path"sv) {
                train_command.db_path = std::string(value);

            } else if (name == "path_to_save"sv) {
                train_command.path_to_save = std::string(value);

            } else if (name == "cycles"sv) {
                int cycles = StringViewToInt(value);
                if (cycles < 1) {
                    throw std::invalid_argument("Number of cycles must be greater than 0"s);
                }
                train_command.training_cycles = cycles;

            } else if (name == "algorithm"sv) {
                int algorithm = StringViewToInt(value);
                if (algorithm < RequestHandler::SEQUENTIALLY
                    || algorithm > RequestHandler::SHUFFLED_WITH_NOT_SYM) {
                    throw std::invalid_argument("Only algorithms 0 (sequential), "
                        "1 (shuffled), 2 (shuffled_with_not_sym) are supported"s);
                }
                train_command.algorithm = static_cast<RequestHandler::Algorithm>(algorithm);

            } else {
                throw ParsingError("Unsupported parameter '"s
                        + std::string(name) + "'"s);
            }
        }
        command = train_command;

    } else if (name == "recognize"sv) {
        RecognizeCommand recognize_command;
        for (int i = 1; i < strings.size(); ++i) {
            std::string_view str = strings[i];
            auto [name, value] = ParseParameter(str);

            if (name == "snn_data_path"sv) { 
                recognize_command.snn_data_path = std::string(value);

            } else if (name == "target_path"sv) {
                recognize_command.target_path = std::string(value);

            } else if (name == "result_path"sv) {
                recognize_command.result_path = std::string(value);

            } else {
                throw ParsingError("Unsupported parameter '"s
                        + std::string(name) + "'"s);
            }
        }
        command = recognize_command;

    } else {
        throw ParsingError("Unsupported command '"s
                + std::string(name) + "'"s);
    }

    return command;
}

const std::string help_string = 
    "Commands\n\n"
    "1. train - Creates a new neural network and trains it with images\n"
    "from the folder.\n\n"
    "Options:\n"
    "    -snn_data_path - Path to the pre-trained neural network.\n"
    "    Default value is an empty string, which creates an untrained\n"
    "    neural network.\n\n"
    "    -db_path - Path to the folder with images. Default value\n"
    "    is \"training_chars\".\n\n"
    "    -path_to_save= - Path to save the trained neural network data.\n"
    "    Default value is \"snn_data\".\n\n"
    "    -cycles - Number of training cycles. Default value is 1000.\n\n"
    "    -algorithm - Training algorithm. Default value is 1. Algorithms\n"
    "     0 (sequential), 1 (shuffled), 2 (shuffled_with_not_sym) are supported.\n\n"
    "2. recognize - Loads the neural network data and recognizes an image or\n"
    "a folder with images.\n\n"
    "Options:\n"
    "    -snn_data_path - Path to the pre-trained neural network. Default value\n"
    "    is \"snn_data\".\n\n"
    "    -target_path - Path to the image file or folder with images for\n"
    "    recognition. Default value is \"target_chars\".\n\n"
    "    -result_path - Path to save the report as a text file. If not specified,\n"
    "    the report will be displayed in the terminal. Default value is\n"
    "    an empty string."s;

void InterpretCommand(Command command) {
    RequestHandler handler;

    if (std::holds_alternative<HelpCommand>(command)) {
        HelpCommand help_command = std::get<HelpCommand>(command);
        std::cout << help_string << std::endl;
        
    } else if (std::holds_alternative<TrainCommand>(command)) {
        TrainCommand train_command = std::get<TrainCommand>(command);
        if (train_command.snn_data_path.empty()) {
            handler.CreateNewSnn();
        } else {
            handler.LoadSnn(train_command.snn_data_path);
        }

        handler.LoadDb(train_command.db_path);
        handler.SetAlgorithm(train_command.algorithm);
        handler.Train(train_command.training_cycles, std::cout);
        handler.SaveSnn(train_command.path_to_save);

    } else if (std::holds_alternative<RecognizeCommand>(command)) {
        RecognizeCommand recogn_command = std::get<RecognizeCommand>(command);
        handler.LoadSnn(recogn_command.snn_data_path);

        std::ostream *os;
        std::ofstream result_file; 
        if (recogn_command.result_path.empty()) {
            os = &std::cout;
        } else {
            result_file.open(recogn_command.result_path);
            if (!result_file) {
                throw std::runtime_error("Unable to open file "s
                    + recogn_command.result_path + " for saving result"s);
            }
            os = &result_file;
        }
        handler.Recognize(recogn_command.target_path, *os);
        
 
    } else {
        std::cout << "Unrealized command"s << std::endl;
    }
}

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

    try {
        Command command = ParseStrings(strings);
        InterpretCommand(std::move(command));
    } catch (const ParsingError& e) {
        std::cout << "A parsing error has occurred: "s + e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "An error has occurred: "s + e.what() << std::endl;
    }
}