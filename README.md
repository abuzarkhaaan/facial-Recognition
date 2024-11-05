# Facial Recognition System

A facial recognition system that can enroll and recognize faces using a webcam or image files. This project leverages modern deep learning tools such as InsightFace for face detection and ArcFace for face embedding extraction. A simple graphical user interface (GUI) is also provided using `Tkinter` to easily add new faces and recognize faces from captured images. Additionally, this system is designed to handle occluded faces, such as those wearing masks, partially visible faces, or faces covered by other objects.

The aim of this project is to provide an easy-to-use, user-friendly interface that allows individuals to explore the possibilities of face recognition technology. Whether you are a student, developer, or enthusiast, this project provides a comprehensive solution to learn about facial recognition using some of the most popular deep learning frameworks available today. You can quickly register new faces in the system and then recognize them from a live feed or image files, making this project suitable for practical applications like attendance systems, door access management, and more.

## Features
- **Add New Face:** Register a new person using a webcam or from an image file, with an easy-to-follow workflow that guides the user through the process.
- **Face Recognition:** Recognize faces from live webcam input or an image file. This feature uses advanced facial recognition techniques to determine if the face belongs to an already known individual.
- **Face Detection with Occlusion Handling:** Utilize the InsightFace detector to accurately locate faces in an image, even when there is occlusion. InsightFace is known for its robustness, enabling the system to handle diverse facial poses, angles, and occlusions, such as:
  - **Half Face Visibility:** The system can detect and recognize faces that are only partially visible, such as those at the edge of the frame.
  - **Wearing Masks:** The system can effectively detect and process faces even if the individual is wearing a mask or other face coverings, making it suitable for real-world scenarios where mask-wearing is common.
  - **Other Occlusions:** Handles faces covered by hats, sunglasses, hands, or other items.
- **Embeddings with ArcFace:** Compute face embeddings with a pre-trained ArcFace model. ArcFace produces high-quality embeddings that are used to uniquely identify each individual. This embedding approach ensures that different images of the same person map to similar vectors.

The system is designed to work seamlessly and efficiently, allowing for high accuracy in face recognition tasks. All of these features are integrated into a convenient GUI, so users can easily interact with the software without requiring prior programming knowledge.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [How It Works](#how-it-works)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Future Improvements](#future-improvements)

## Installation
To get started with the project, follow these instructions:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Musawer1214/Facial-Recognition.git
    cd Facial-Recognition
    ```

2. **Install Dependencies**:
    It is recommended to create a virtual environment before installing the dependencies to avoid conflicts with other Python packages you might have installed.
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Download Pre-trained Models**:
   - This project uses the pre-trained weights for the ArcFace model from `insightface`. These weights will be downloaded automatically when the model is first used, making setup simpler for users.

## Usage
Run the script to launch the graphical interface for face recognition:

```sh
python main.py
```

The GUI is designed with simplicity in mind, allowing users to perform tasks without any prior experience in computer vision. Once you run the script, you will be presented with a simple interface that provides access to all functionalities.

### GUI Features
- **Add New Face**: Click the `Add New Face` button to add a new person to the database. You can choose to use either a saved image file or capture a photo using your webcam. The process involves detecting the face, generating an embedding, and saving the embedding in the database.
- **Recognize Face**: Click the `Recognize Face` button to recognize a face from an image or live webcam input. This feature will detect any faces in the provided image, calculate embeddings for each detected face, and match them against the stored embeddings to identify known individuals.
- **Occlusion Handling**: The system will prompt if a detected face is partially occluded but still recognizable, providing insights even under challenging conditions such as mask-wearing or partial coverage.

These features help build a practical face recognition system that can be used in various real-world scenarios. The GUI makes it easy to understand and utilize face recognition even if you are a beginner in the field.

## Dependencies
The following Python libraries are used in this project:
- `opencv-python` for image processing and capturing webcam input.
- `numpy` for numerical operations, including embedding calculations.
- `torch` and `torchvision` for working with deep learning models and managing data transformations.
- `insightface` for detecting faces in images and extracting face embeddings, providing a reliable solution for face localization even when faces are partially occluded.
- `Pillow` for image handling and processing, making it easier to work with different image formats.
- `Tkinter` for creating the graphical user interface, providing an interactive way to use the face recognition system.

You can install all dependencies by running:

```sh
pip install -r requirements.txt
```

The `requirements.txt` file should contain:
```
opencv-python
numpy
torch
insightface
Pillow
tk
```

These dependencies are necessary for running the project smoothly. Ensure that all dependencies are properly installed to avoid runtime issues.

## How It Works
- **Face Detection**: The program uses InsightFace to detect faces in an image or video stream. InsightFace can detect multiple faces simultaneously, and its use of facial landmarks makes it suitable for capturing faces from various angles and lighting conditions, even with partial occlusion.
- **Face Embedding Extraction**: The detected faces are resized to a standard size and passed through the ArcFace model to obtain 512-dimensional embeddings. These embeddings are unique representations of each face, enabling effective comparisons between different faces.
- **Face Recognition**: To recognize a face, the embedding of the new face is compared to the embeddings stored in the database using cosine similarity. A threshold value is set to determine whether the face matches a known identity. If the similarity is above this threshold, the system will recognize the face as belonging to a specific individual, even in cases of partial occlusion.
- **Occlusion Handling**: The system's robust detection mechanism handles occluded faces (e.g., masks, sunglasses, or partially visible faces), ensuring recognition accuracy even in challenging scenarios.
- **GUI**: A user-friendly GUI built with `Tkinter` allows users to add and recognize faces conveniently. Users can interact with the application by clicking buttons, capturing photos, and receiving feedback messages, making the experience seamless and enjoyable.

## File Structure
```
facial-recognition/
├── main.py                  # Main file to run the application
├── embedding_database_arcface.pkl   # Stores known face embeddings
├── requirements.txt         # List of dependencies
└── README.md                # Documentation (this file)
```
- **`main.py`**: Contains the core logic for face detection, face embedding extraction, and GUI implementation. This script serves as the entry point to the application.
- **`embedding_database_arcface.pkl`**: A serialized dictionary that stores face embeddings in a key-value pair format, where the key is the name of the individual and the value is the corresponding embedding vector.
- **`requirements.txt`**: Lists all the dependencies required for the project, ensuring that users can set up their environment quickly and without hassle.

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. **Fork the repository**: This will create a copy of the repository in your GitHub account.
2. **Create a new branch**: Create a feature branch to work on a new feature or address a bug (`git checkout -b feature-name`).
3. **Commit your changes**: Make changes and commit them with a meaningful commit message (`git commit -m 'Add feature'`).
4. **Push to the branch**: Push your changes to your forked repository (`git push origin feature-name`).
5. **Open a pull request**: Create a pull request to merge your changes into the main repository.

Before contributing, please make sure to check the existing issues and follow the coding guidelines to ensure consistency across the project. We appreciate your help in improving the Facial Recognition System.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute this project under the terms of the MIT License.

## Acknowledgments
- The `insightface` library provides a comprehensive toolkit for face detection and embedding extraction, making it straightforward to implement face recognition systems.
- Thanks to the contributors of the `Tkinter` community for making GUI creation accessible and straightforward. Tkinter's ease of use allowed us to create a fully functional graphical user interface that caters to both beginners and more advanced users.
- Special thanks to the open-source community for providing the resources and tools that make projects like this possible. The combination of high-quality models and open-source software has enabled us to develop a facial recognition system that is both practical and educational.

## Future Improvements
This project aims to evolve with contributions and suggestions from the community. Some ideas for future enhancements include:
- **Database Management Features**: Add features to manage the database, such as deleting or updating existing records.
- **Improved GUI**: Enhance the GUI with more modern design elements, user guidance, and support for additional features.

Your ideas and contributions can make this project even more useful, robust, and user-friendly!
