#pragma once

#include "snn.h"
#include "training_database.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>

class RequestHandler {
public:
    void CreateNewSnn();
    void LoadSnn(const std::filesystem::path& snn_data_path);
    void SaveSnn(const std::filesystem::path& path_to_save) const;
    void LoadDb(const std::filesystem::path& db_path);
    void TrainSequentially(int cycles, std::ostream& progress_output);

    void Recognize(const std::filesystem::path& target_path, std::ostream& output);

private:
    std::unique_ptr<ImageFileNormalizer> normalizer_;
    std::unique_ptr<TrainingDatabase> db_;
    std::unique_ptr<Snn> snn_;

    void RecognizeFolder(const std::filesystem::path& target_path, std::ostream& output, int* n);
    void RecognizeImage(const std::filesystem::path& target_path, std::ostream& output, const int* n);
};