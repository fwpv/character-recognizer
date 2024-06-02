#include "request_handler.h"
#include "profiler.h"
#include "snn.h"
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

    const TrainingDatabase::CharsDict& dict = db_->GetCharsDictionary();
    for (auto& it : dict) {
        char c = it.first;
        if (c < '0' || c > '9') {
            throw std::runtime_error("Char "s + c + " is not supported"s);
        }
    }
}

void RequestHandler::SetAlgorithm(Algorithm algorithm) {
    algorithm_ = algorithm;
}

void RequestHandler::Train(int cycles, std::ostream& progress_output) {
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

    TrainingDatabase::CharPtrArray char_ptr_array = db_->CreateCharPtrArray();
    progress_output << 0;
    for (int i = 0; i < cycles; ++i) {
        if (algorithm_ == SHUFFLED) {
            TrainingDatabase::ShuffleCharPtrArray(char_ptr_array);
        }
        for (auto [c, vec_ptr] : char_ptr_array) {
            size_t number = c - '0';
            snn_->CalculateOutput(*vec_ptr);
            set_unit(resources, number);
            snn_->PropagateErrorBack(resources);
        }
        progress_output << '\r';
        progress_output << i + 1;
    }
    progress_output << std::endl;
}

void RequestHandler::Recognize(const std::filesystem::path& target_path, std::ostream& output) {
    assert(snn_);

    if (!exists(target_path)) {
        throw std::runtime_error(target_path.string() + " does not exist"s);
    }
    normalizer_ = std::make_unique<ImageFileNormalizer>(1024);
    if (is_directory(target_path)) {
        if (is_empty(target_path)) {
            throw std::runtime_error(target_path.string() + " is empty"s);
        }
        file_counter_ = 1;
        RecognizeFolder(target_path, output);
    } else if (is_regular_file(target_path)) {
        file_counter_ = -1;
        RecognizeImage(target_path, output);
    }
}

void RequestHandler::RecognizeFolder(const std::filesystem::path& target_path, std::ostream& output) {
    output << std::endl << "Folder: "s << target_path << std::endl;
    for (const auto& sub : directory_iterator(target_path)) {  
        if (sub.is_directory()) {
            RecognizeFolder(sub.path(), output); // Call recursively
        } else if (sub.is_regular_file()) {
            RecognizeImage(sub.path(), output);
            ++file_counter_;
        }
    }
}

void RequestHandler::RecognizeImage(const std::filesystem::path& target_path, std::ostream& output) {
    if (file_counter_ != -1) {
        output << std::endl << file_counter_ << ". "s;
    }
    output << target_path.string() << std::endl;

    auto vec = normalizer_->Load(target_path);
    snn_->CalculateOutput(vec);
    const std::vector<float>& snn_out = snn_->ReadOutput();
    auto it = std::max_element(snn_out.begin(), snn_out.end());
    size_t max = it - snn_out.begin();
    // Only if the input value is greater than 0.5, the character is considered recognized
    if (*it > 0.50f) { 
        output << "Recognized: "s << max << std::endl;
    } else {
        output << "Closer to: "s << max << std::endl;
    }
    output << "Snn output: "s;
    output << std::fixed << std::showpoint << std::setprecision(3);
    for (int i = 0; i < 10; ++i) {
        if (i > 0) {
            output << ", "s;
        }
        output << snn_out[i];
    }
    output << std::endl;
}