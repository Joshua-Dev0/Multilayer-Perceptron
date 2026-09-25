
# Multilayer Perceptron

A handwritten digit recognition project built using a manually implemented Multilayer Perceptron (MLP) neural network.

The model can recognize handwritten digits from **0 to 9** using the MNIST dataset. It includes a Tkinter canvas where users can draw digits and run inference using the trained model.

## Features

- Handwritten digit recognition (0–9)
- Trained using the MNIST dataset
- Interactive 28 × 28 pixel drawing canvas
- GPU-accelerated computation using CuPy
- Manually implemented forward propagation
- ReLU activation functions for hidden layers
- Softmax activation for digit classification
- Model saving and loading using NumPy-compatible `.npz` files

## Neural Network Architecture

The neural network uses a fully connected Multilayer Perceptron with the following architecture:

```text
Input Layer:   784 neurons
      |
      v
Hidden Layer 1: 64 neurons
      |
     ReLU
      |
      v
Hidden Layer 2: 64 neurons
      |
     ReLU
      |
      v
Output Layer: 10 neurons
      |
    Softmax
      |
      v
Predicted Digit: 0–9
```

### Input

Each MNIST image is a 28 × 28 grayscale image.

The image is flattened into a vector containing 784 pixel values and normalized from 0–255 to 0–1.

### Hidden Layers

The network contains two fully connected hidden layers, each with 64 neurons.

Both hidden layers use the ReLU activation function.

### Output Layer

The output layer contains 10 neurons, representing the digits 0 through 9.

Softmax converts the output values into probabilities. The digit with the highest probability is selected as the prediction.

## Dataset

This project uses the MNIST handwritten digit dataset in CSV format.

Dataset source:

[Kaggle - MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

The dataset contains grayscale handwritten digit images and their corresponding labels.

Download the dataset and place the following files in the project directory:

```text
mnist_train.csv
mnist_test.csv
```

## Requirements

- Python 3
- NumPy
- CuPy
- Polars
- Matplotlib
- tqdm
- Tkinter

CuPy requires a compatible NVIDIA GPU and CUDA environment.

Install the Python dependencies:

```bash
pip install numpy polars matplotlib tqdm
```

Install the appropriate CuPy package for your CUDA environment. For example:

```bash
pip install cupy-cuda12x
```

Choose the CuPy package that matches your installed CUDA environment. See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

Tkinter is included with many Python installations. On Linux, it may need to be installed separately.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Multilayer-Perceptron.git
```

Navigate to the project directory:

```bash
cd Multilayer-Perceptron
```

Install the dependencies and place the MNIST CSV files in the project directory.

## Usage

The trained model is saved as:

```text
model_55K.npz
```

### Run the Digit Recognizer

After training and saving the model, execute:

```bash
python canvas.py
```

The application opens a 28 × 28 drawing canvas.

1. Draw a handwritten digit using your mouse.
2. Click **Inference** to classify the digit.
3. View the predicted digit and its confidence.
4. Click **Clear Canvas** to draw another digit.

The application uses the saved model to perform inference.

## Project Structure

```text
Multilayer-Perceptron/
│
├── canvas.py
├── multilayer_perceptrion3.py
│
├── model_55K.npz
│
└── README.md
```

| File | Description |
|------|-------------|
| `canvas.py` | Tkinter interface for drawing and recognizing digits |
| `multilayer_perceptrion3.py` | Neural network implementation and training |
| `model_55K.npz` | Saved neural network parameters |

## Implementation Notes

The neural network's forward propagation and backpropagation are implemented manually using matrix operations.

CuPy is used to accelerate numerical computation on the GPU.

The drawing canvas is built using Tkinter, providing an interface for entering handwritten digits.

The canvas is AI-generated, while the neural network's forward propagation is manually implemented.

## Limitations

- The model is designed specifically for handwritten digits from 0 to 9.
- Recognition accuracy depends on the quality of the training and the input handwriting.
- The model expects a 28 × 28 grayscale image.
- The model may produce incorrect predictions for poorly centered or unusually written digits.
- GPU acceleration requires a compatible CUDA environment.

## Future Improvements

- Display probabilities for all 10 digits.
- Improve the drawing brush and image preprocessing.
- Add model accuracy visualization.
- Support batch inference.
- Experiment with different network architectures.
- Compare the MLP with a Convolutional Neural Network (CNN).

## License

Add your preferred open-source license here.

README.md is ai-generated
