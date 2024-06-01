# Character Recognizer

**Character Recognizer** is an experimental program that explores the capabilities of a simple neural network for character recognition. Currently, it only supports digits from 0 to 9.

## How to Build

To build the program, you need CMake and a C++ compiler.

1. Clone the repository to your computer.
2. Go to the `character-recognizer/character-recognizer` folder.
3. Create a `build` directory and go to it:
    ```sh
    mkdir build
    cd build
    ```
4. Run CMake. Example for Windows and MinGW compiler:
    ```sh
    cmake ../ -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
    cmake --build .
    ```
    Alternatively, you can build in VSCode with the CMake Tools extension or in any other development environment that supports CMake.

## Resource Requirements

- All paths must contain only English characters.
- Images for training and recognition: BMP format, 8 or 24 bits per pixel, height 32, width 32.
- The training images folder should contain subfolders named from '0' to '9' (these are the names of the recognizable characters). Each subfolder should contain BMP images sized 32x32. The names and number of images in each subfolder can be any.

## Commands

### 1. `train`
Creates a new neural network and trains it with images from the folder.

**Options:**
- `-snn_data_path="..."` - Path to the pre-trained neural network. Default value is an empty string, which creates an untrained neural network.
- `-db_path="..."` - Path to the folder with images. Default value is `"training_chars"`.
- `-path_to_save="..."` - Path to save the trained neural network data. Default value is `"snn_data"`.
- `-cycles=...` - Number of training cycles. Default value is `1000`.
- `-algorithm=...` - Training algorithm. Default value is `0`. Currently, only one mode is available - sequential.

**Example:**
```sh
recognizer.exe train -db_path="training_chars" -path_to_save="snn_data_2000_sequent" -cycles=2000
```

### 2. `recognize`
Loads the neural network data and recognizes an image or a folder with images.

**Options:**
- `-snn_data_path="..."` - Path to the pre-trained neural network. Default value is `"snn_data"`.
- `-target_path="..."` - Path to the image file or folder with images for recognition. Default value is `"target_chars"`.
- `-result_path="..."` - Path to save the report as a text file. If not specified, the report will be displayed in the terminal. Default value is an empty string.

**Example:**
```sh
recognizer.exe recognize -snn_data_path="snn_data_2000_sequent" -target_path="target_chars" -result_path="result.txt"
```

### 3. `help`
Displays help information about commands and their parameters.
