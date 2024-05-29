#include "command_interpreter.h"

#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

std::pair<std::string_view, std::string_view>
        ParseParameter(std::string_view parameter) {
    size_t equal_pos = parameter.find_first_of('=');
    if (equal_pos == std::string_view::npos) {
        throw std::runtime_error("Parsing error: there is no '=' sign after parameter"s);
    }
    if (equal_pos == 0) {
        throw std::runtime_error("Parsing error: empty string before '=' sign"s);
    }
    if (equal_pos == parameter.size() - 1) {
        throw std::runtime_error("Parsing error: empty string after '=' sign"s);
    }

    std::string_view name = parameter.substr(0, equal_pos);

    // Remove minus
    if (name[0] != '-') {
        throw std::runtime_error("Parsing error: Parameter must start with '-'"s);
    }
    name.remove_prefix(1);
    if (name.empty()) {
        throw std::runtime_error("Parsing error: An empty parameter"s);
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
        throw std::runtime_error("Parsing error: empty string after '=' sign"s);
    }

    return {name, value};
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
            if (name == "db_path"sv) {
                train_command.db_path = std::string(value);
            } else if (name == "path_to_save"sv) {
                train_command.path_to_save = std::string(value);
            } else {
                throw std::runtime_error("Parsing error: unsupported parameter '"s
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
            } else {
                throw std::runtime_error("Parsing error: unsupported parameter '"s
                        + std::string(name) + "'"s);
            }
        }
        command = recognize_command;

    } else {
        throw std::runtime_error("Parsing error: unsupported command '"s
                + std::string(name) + "'"s);
    }

    return command;
}

void InterpretCommand(Command command) {
    std::cout << "Interpret command. Test 1: Check parsing! "s << std::endl;;

    if (std::holds_alternative<HelpCommand>(command)) {
        HelpCommand help_command = std::get<HelpCommand>(command);
        std::cout << "Command: help"s << std::endl;
        
    } else if (std::holds_alternative<TrainCommand>(command)) {
        TrainCommand train_command = std::get<TrainCommand>(command);
        std::cout << "Command: train"s << std::endl;
        std::cout << "With parameters: "s << std::endl;
        std::cout << "db_path: "s << train_command.db_path << std::endl;
        std::cout << "path_to_save: "s << train_command.path_to_save << std::endl;
        std::cout << "training_cycles: "s << train_command.training_cycles << std::endl;
        std::cout << "algorithm: "s << train_command.algorithm << std::endl;

    } else if (std::holds_alternative<RecognizeCommand>(command)) {
        RecognizeCommand recogn_command = std::get<RecognizeCommand>(command);
        std::cout << "Command: recognize"s << std::endl;
        std::cout << "With parameters: "s << std::endl;
        std::cout << "snn_data_path: "s << recogn_command.snn_data_path << std::endl;
        std::cout << "target_path: "s << recogn_command.target_path << std::endl;
        std::cout << "result_path: "s << recogn_command.result_path << std::endl;
 
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
        InterpretCommand(command);
    } catch (const std::exception& e) {
        std::cout << "An error occurred: "s + e.what() << std::endl;
    }
}