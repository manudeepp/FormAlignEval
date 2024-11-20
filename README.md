# Form Alignment Evaluation

This repository contains code for evaluating the effectiveness of different image alignment algorithms, including ORB, SIFT, FREAK, and a combined FREAK+SIFT approach. The project is designed to process forms and assess how well each algorithm aligns the scanned images with a template.

## Features

- **Multiple Alignment Algorithms**: The code supports ORB, SIFT, FREAK, and a hybrid FREAK+SIFT method for image alignment.
- **Dynamic Feature Matching**: Each algorithm is tested with varying numbers of keypoints to find the optimal balance between speed and accuracy.
- **Automated PDF Reporting**: Results, including alignment success rates, processing times, and matching statistics, are compiled into a comprehensive PDF report.
- **Customizable Keypoints**: Users can specify keypoints and check phrases via a CSV file to fine-tune the alignment verification process.

## Getting Started

### Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- OpenCV
- matplotlib
- pandas
- pytesseract
- pdf2image
- numpy

You can install the necessary Python packages with:

```bash
pip install -r requirements.txt
```
### Project Structure

- **pdfInputs/**: Folder containing input PDF files.
- **outputImages/**: Folder where the converted PNG files are saved.
- **results/**: Folder where the PDF report is saved.
- **keypointsLocations.csv**: CSV file containing keypoint coordinates and check phrases.
- **template.png**: Template image used for alignment.
- **align_eval.py**: Main Python script to run the evaluation.
- **README.md**: This README file.


### Running the Script

1. **Place Your PDFs**: Add the PDF forms you want to process into the `pdfInputs/` folder.

2. **Run the Script**: Execute the `align_eval.py` script.

   - **Default mode (debugging off)**:
   
     ```bash
     python align_eval.py
     ```

   - **Debug mode (debugging on)**:
   
     ```bash
     python align_eval.py --debug
     ```

3. **View Results**: The results will be saved in the `results/` folder as a PDF report.

### Test Set

The repository includes a basic test set to help you get started. However, to thoroughly evaluate the algorithms under various conditions, we encourage you to create your own test set:

1. **Use the Provided Template**: Start with the `template.png` image provided in this repository.

2. **Introduce Distortions**: Apply different transformations to the template image, such as rotations, scaling, translations, perspective changes, and noise. This will allow you to test the robustness of each alignment algorithm.

3. **Run the Evaluation**: Process the distorted images with the provided script to see how well the algorithms perform under various conditions.

### Encouragement for Exploration

This project is just the starting point. I encourage users to experiment with the template image and create their own test sets by introducing various distortions. This will help you to explore the strengths and weaknesses of different alignment algorithms in a more comprehensive manner.

---

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - feel free to use, modify, and distribute it as long as proper credit is given to the original author and contributors.
