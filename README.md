# Sketchify

Sketchify is a Streamlit application that converts photos into black and white outline sketches suitable for printing and coloring.

## Features

- Upload images or capture them using your webcam
- Convert images to black outline sketches with a white background
- Adjust the level of detail in the outlines (from easy to expert)
- Download the converted sketches for printing

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   streamlit run app.py
   ```
2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)
3. Upload an image or take a photo with your webcam
4. Adjust the detail level using the slider
5. Download the converted sketch

## How It Works

Sketchify uses computer vision techniques including:
- Edge detection algorithms
- Convolutional image filtering
- Adaptive thresholding

These techniques are combined to extract the outlines from images while allowing users to control the level of detail. 