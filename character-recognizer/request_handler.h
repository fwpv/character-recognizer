#pragma once

#include "snn.h"
#include "training_database.h"

#include <filesystem>
#include <iostream>
#include <memory>

class RequestHandler {
public:
    void CreateNewSnn();
    void LoadSnn(const std::filesystem::path& snn_data_path);
    void SaveSnn(const std::filesystem::path& path_to_save);
    void LoadDb(const std::filesystem::path& db_path);
    void Train();

    void Recognize(const std::filesystem::path& target_path, std::ostream& out);

private:
    std::unique_ptr<ImageFileNormalizer> normalizer_;
    std::unique_ptr<TrainingDatabase> db_;
    std::unique_ptr<Snn> snn_;
};

void RunExperiment1();