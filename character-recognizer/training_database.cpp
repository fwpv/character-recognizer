#include "training_database.h"
#include "bmp_image.h"

#include <exception>

using namespace std::literals;

ImageFileNormalizer::ImageFileNormalizer(size_t input_width)
: input_width_(input_width) {
}

std::vector<float> ImageFileNormalizer::Load(const std::filesystem::path& file) const {
    img_lib::Image image = img_lib::LoadBMP(file);
    const int w = image.GetWidth();
    const int h = image.GetHeight();

    if (w == 0) {
        throw std::runtime_error("The file "s + file.string() + " cannot be read"s);
    }
    if (w != h) {
        throw std::runtime_error("The sides of the image must be the same"s);
    }
    if (input_width_ != static_cast<size_t>(w * h)) {
        throw std::runtime_error("Pixel count does not match the input width of the network"s);
    }

    auto normalize_color = [](const img_lib::Color& color) {
        uint32_t uval = 0;
        uval = static_cast<uint32_t>(color.r) << 24
               | static_cast<uint32_t>(color.g) << 16
               | static_cast<uint32_t>(color.b) << 8;

        return static_cast<float>(uval) / UINT32_MAX;
    };

    std::vector<float> vec(input_width_);
    for (int y = 0; y < h; ++y) {
        const img_lib::Color* line = image.GetLine(y);
        for (int x = 0; x < w; ++x) {
            vec[y * w + x] = normalize_color(line[x]);
        }
    }

    return vec;
}

TrainingDatabase::TrainingDatabase(const FileNormalizerInterface* file_normalizer)
: file_normalizer_(file_normalizer) {
}

void TrainingDatabase::BuildFromFolder(const std::filesystem::path& folder) {
    using namespace std::filesystem;
    if (!exists(folder)) {
        throw std::runtime_error(folder.string() + " does not exist"s);
    }
    if (!is_directory(folder)) {
        throw std::runtime_error(folder.string() + " is not a directory"s);
    }
    if (is_empty(folder)) {
         throw std::runtime_error(folder.string() + " is empty"s);
    }
    for (const auto& sub : directory_iterator(folder)) {
        if (sub.is_directory()) {
            std::string name = sub.path().filename().string();
            if (is_empty(sub)) {
                continue;
            }
            if (name.size() == 1) {
                Data& data = data_dict_[name[0]];
                for (const auto& file : directory_iterator(sub)) {
                    data.push_back(file_normalizer_->Load(file.path()));
                }
            } else { // not symbols
                for (const auto& file : directory_iterator(sub)) {
                    non_char_data_.push_back(file_normalizer_->Load(file.path()));
                }
            }
        }
    }
}

const TrainingDatabase::DataDict& TrainingDatabase::GetDataDictionary() const {
    return data_dict_;
}

const TrainingDatabase::Data& TrainingDatabase::GetNonCharData() const {
    return non_char_data_;
}

std::vector<char> TrainingDatabase::GetUploadedChars() const {
    std::vector<char> chars;
    chars.reserve(data_dict_.size());
    for (auto it : data_dict_) {
        chars.push_back(it.first);
    }
    return chars;
}