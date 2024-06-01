#include "snn.h"
#include "profiler.h"
#include "request_handler.h"
#include "state_saver.h"
#include "training_database.h"

#include <algorithm>
#include <cassert>
#include <conio.h> 
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace std::filesystem;
using namespace std::literals;

void RequestHandler::CreateNewSnn() {
    snn_ = std::make_unique<Snn>(1024, 2, 1024, 10);
    snn_->InitializeBiasesWithRandom();
    snn_->InitializeWeightsWithRandom();
    snn_->SetLearningCoefficient(0.1f);
}

void RequestHandler::LoadSnn(const path& snn_data_path) {
    SnnMemento snn_state = LoadSnnState(snn_data_path);
    snn_ = std::make_unique<Snn>(snn_state);
}

void RequestHandler::SaveSnn(const path& path_to_save) const {
    assert(snn_);
    SnnMemento snn_state = snn_->CreateMemento();
    SaveSnnState(path_to_save, snn_state);
}

void RequestHandler::LoadDb(const path& db_path) {
    normalizer_ = std::make_unique<ImageFileNormalizer>(1024);
    db_ = std::make_unique<TrainingDatabase>(normalizer_.get());
    db_->BuildFromFolder(db_path);

    const TrainingDatabase::DataDict& dict = db_->GetDataDictionary();
    for (auto& it : dict) {
        char c = it.first;
        if (c < '0' || c > '9') {
            throw std::runtime_error("Char "s + c + " is not supported"s);
        }
    }
}

void RequestHandler::TrainSequentially(int cycles, std::ostream& progress_output) {
    assert(db_);
    assert(snn_);

    auto set_unit = [](std::vector<float>& vec, size_t pos) {
        std::fill(vec.begin(), vec.end(), 0.0f);
        vec[pos] = 1;
    };

    auto clear = [](std::vector<float>& vec) {
        std::fill(vec.begin(), vec.end(), 0.0f);
    };

    std::vector<float> resources(10, 0.0f);

    const TrainingDatabase::DataDict& dict = db_->GetDataDictionary();
    progress_output << 0;
    for (int i = 0; i < cycles; ++i) {
        for (auto& it : dict) {
            char c = it.first;
            size_t number = c - '0';
            for (auto& vec : it.second) {
                snn_->CalculateOutput(vec);
                set_unit(resources, number);
                snn_->PropagateErrorBack(resources);
            }
        }
        progress_output << '\r';
        progress_output << i + 1;
    }
    progress_output << std::endl;
}

void RequestHandler::Recognize(const path& target_path, std::ostream& output) {
    assert(snn_);

    if (!exists(target_path)) {
        throw std::runtime_error(target_path.string() + " does not exist"s);
    }
    normalizer_ = std::make_unique<ImageFileNormalizer>(1024);
    if (is_directory(target_path)) {
        if (is_empty(target_path)) {
            throw std::runtime_error(target_path.string() + " is empty"s);
        }
        throw std::runtime_error("Folder content recognition is not supported yet"s);
    } else if (is_regular_file(target_path)) {
        RecognizeImage(target_path, output);
    }
}

void RequestHandler::RecognizeImage(const path& target_path, std::ostream& output) {
    std::cout << "File: " << target_path.string() << std::endl;
    auto vec = normalizer_->Load(target_path);
    snn_->CalculateOutput(vec);
    const std::vector<float>& snn_out = snn_->ReadOutput();
    auto it = std::max_element(snn_out.begin(), snn_out.end());
    size_t max = it - snn_out.begin();
    output << "Recognized: "s << max << std::endl;
    output << "Precision: "s;
    std::cout << std::fixed << std::showpoint << std::setprecision(3);
    bool first = true;
    for (int i = 0; i < 10; ++i) {
        if (!first) {
            output << ", "s;
        }
        first = false;
        output << static_cast<char>('0' + i) << ": "s << snn_out[i];
    }
    output << std::endl;
}