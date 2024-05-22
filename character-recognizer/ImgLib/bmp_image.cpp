#include "bmp_image.h"
#include "pack_defines.h"

#include <array>
#include <fstream>
#include <string_view>
#include <stdint.h>

using namespace std;

namespace img_lib {

PACKED_STRUCT_BEGIN BitmapFileHeader {
    uint8_t signature[2]; // символы подписи
    uint32_t total_size;
    uint32_t reserve;
    uint32_t indentation;
}
PACKED_STRUCT_END

PACKED_STRUCT_BEGIN BitmapInfoHeader {
    uint32_t header_size;
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t bpp;
    uint32_t compression_type;
    uint32_t data_size;
    uint32_t hres;
    uint32_t vres;
    uint32_t used_colors;
    uint32_t colors;
}
PACKED_STRUCT_END

PACKED_STRUCT_BEGIN RgbQuad {
    uint8_t b;
    uint8_t g;
    uint8_t r;
    uint8_t reserved;
}
PACKED_STRUCT_END

// функция вычисления отступа по ширине
static int GetBMPStride(int w, uint16_t bpp = 24) {
    assert(bpp == 24 || bpp == 8);
    if (bpp == 24) {
        return 4 * ((w * 3 + 3) / 4);
    } else {
        return 4 * ((w + 3) / 4);
    }
}

bool SaveBMP(const Path& file, const Image& image) {
    ofstream out(file, ios::binary);

    const int w = image.GetWidth();
    const int h = image.GetHeight();
    int stride = GetBMPStride(w);

    BitmapFileHeader file_header;
    file_header.signature[0] = 'B';
    file_header.signature[1] = 'M';
    file_header.total_size = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader) + stride * h;
    file_header.reserve = 0;
    file_header.indentation = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader);

    // Записать BitmapFileHeader
    out.write(reinterpret_cast<const char*>(&file_header), sizeof(BitmapFileHeader));

    BitmapInfoHeader info_header;
    info_header.header_size = sizeof(BitmapInfoHeader);
    info_header.width = w;
    info_header.height = h;
    info_header.planes = 1;
    info_header.bpp = 24;
    info_header.compression_type = 0;
    info_header.data_size = stride * h;
    info_header.hres = 11811;
    info_header.vres = 11811;
    info_header.used_colors = 0;
    info_header.colors = 0x1000000;

    // Записать BitmapInfoHeader
    out.write(reinterpret_cast<const char*>(&info_header), sizeof(BitmapInfoHeader));

    // Записать данные
    std::vector<char> buff(stride);
    for (int y = h - 1; y >= 0; --y) {
        const Color* line = image.GetLine(y);
        for (int x = 0; x < w; ++x) {
            buff[x * 3 + 0] = static_cast<char>(line[x].b);
            buff[x * 3 + 1] = static_cast<char>(line[x].g);
            buff[x * 3 + 2] = static_cast<char>(line[x].r);
        }
        std::fill(buff.begin() + w * 3, buff.end(), 0);
        out.write(buff.data(), buff.size());
    }

    return out.good();
}

Image LoadBMP(const Path& file) {
    ifstream ifs(file, ios::binary);

    // Прочитать BitmapFileHeader
    BitmapFileHeader file_header;
    ifs.read(reinterpret_cast<char*>(&file_header), sizeof(BitmapFileHeader));

    // Проверить подпись
    if (file_header.signature[0] != 'B'
        || file_header.signature[1] != 'M') {
        return {};
    }

    // Записать BitmapInfoHeader
    BitmapInfoHeader info_header;
    ifs.read(reinterpret_cast<char*>(&info_header), sizeof(BitmapInfoHeader));

    // Поддерживается формат с 24 или 8 цветами на пиксель, без сжатия
    if (info_header.planes != 1
        || !(info_header.bpp == 8 || info_header.bpp == 24)
        || info_header.compression_type != 0) {
        return {};
    }

    int w = info_header.width;
    int h = info_header.height;
    int stride = GetBMPStride(w, info_header.bpp);

    Image iamge(w, h, Color::Black());
    std::vector<char> buff(stride);

    // Палитра для 8-битных bmp
    std::vector<Color> palette;
    if (info_header.bpp == 8) {
        palette.resize(256);
        for (int i = 0; i < 256; ++i) {
            RgbQuad quad;
            ifs.read(reinterpret_cast<char*>(&quad), sizeof(RgbQuad));
            palette[i] = Color{static_cast<byte>(quad.r),
                               static_cast<byte>(quad.g),
                               static_cast<byte>(quad.b),
                               static_cast<byte>(0)};
        }
    }

    // Прочитать данные
    for (int y = h - 1; y >= 0; --y) {
        Color* line = iamge.GetLine(y);
        ifs.read(buff.data(), buff.size());

        if (info_header.bpp == 24) {
            for (int x = 0; x < w; ++x) {
                line[x].b = static_cast<byte>(buff[x * 3 + 0]);
                line[x].g = static_cast<byte>(buff[x * 3 + 1]);
                line[x].r = static_cast<byte>(buff[x * 3 + 2]);
            }
        } else { // info_header.bpp == 8
            for (int x = 0; x < w; ++x) {
                uint8_t index = static_cast<uint8_t>(buff[x]);
                line[x] = palette[index];
            }
        }
    }

    return iamge;
}
}  // namespace img_lib