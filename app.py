import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def convert_to_sketch(image, detail_level):
    """
    Convert an image to a sketch with adjustable detail level.
    
    Args:
        image: Input image (numpy array)
        detail_level: Level of detail (0-100) where 0 is minimal detail and 100 is maximum detail
    
    Returns:
        Sketch image (numpy array)
    """
    # Convert to grayscale if the image is colored
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    # Invert the grayscale image
    inverted_image = 255 - gray_image
    
    # Apply Gaussian blur
    # Adjust blur based on detail level (less blur for more details)
    blur_value = max(1, int(15 - (detail_level / 100) * 14))
    if blur_value % 2 == 0:  # Ensure blur_value is odd
        blur_value += 1
    
    blurred_image = cv2.GaussianBlur(inverted_image, (blur_value, blur_value), 0)
    
    # Invert the blurred image
    inverted_blurred = 255 - blurred_image
    
    # Create the pencil sketch image by dividing the grayscale image by the inverted blurred image
    sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
    
    # Apply edge detection with adjustable parameters based on detail level
    low_threshold = int(10 + (detail_level / 100) * 90)
    high_threshold = low_threshold * 3
    
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    
    # Combine the sketch and edges based on detail level
    detail_weight = detail_level / 100
    combined = cv2.addWeighted(sketch, 1 - detail_weight, edges, detail_weight, 0)
    
    # Apply adaptive thresholding for more defined outlines
    # Adjust block size based on detail level
    block_size = max(3, int(3 + (detail_level / 100) * 10))
    if block_size % 2 == 0:  # Ensure block_size is odd
        block_size += 1
    
    threshold_image = cv2.adaptiveThreshold(
        combined, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        block_size, 
        2
    )
    
    # Invert the image to get black outlines on white background
    final_sketch = 255 - threshold_image
    
    # Apply morphological operations to clean up the sketch
    kernel_size = max(1, int(1 + (detail_level / 100) * 2))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    final_sketch = cv2.morphologyEx(final_sketch, cv2.MORPH_CLOSE, kernel)
    
    # Invert again to get black outlines on white background
    final_sketch = 255 - final_sketch
    
    return final_sketch

def main():
    st.set_page_config(
        page_title="Sketchify",
        page_icon="✏️",
        layout="wide"
    )
    
    st.title("✏️ Sketchify")
    st.subheader("Convert photos to coloring pages")
    
    # Sidebar for controls
    st.sidebar.title("Controls")
    
    # Detail level slider
    detail_level = st.sidebar.slider(
        "Detail Level", 
        min_value=0, 
        max_value=100, 
        value=50,
        help="Adjust the level of detail in the sketch (Easy: less details, Expert: more details)"
    )
    
    # Display detail level as text
    if detail_level < 33:
        detail_text = "Easy (fewer details)"
    elif detail_level < 66:
        detail_text = "Medium"
    else:
        detail_text = "Expert (more details)"
    
    st.sidebar.text(f"Current level: {detail_text}")
    
    # Image source selection
    image_source = st.sidebar.radio("Image Source", ["Upload Image", "Webcam"])
    
    # Initialize session state for storing the image
    if 'input_image' not in st.session_state:
        st.session_state.input_image = None
    
    if 'sketch_image' not in st.session_state:
        st.session_state.sketch_image = None
    
    # Create two columns for input and output
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input Image")
        
        if image_source == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Read the image
                image = Image.open(uploaded_file)
                st.session_state.input_image = np.array(image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        else:  # Webcam
            st.write("Take a photo with your webcam")
            webcam_image = st.camera_input("Capture")
            
            if webcam_image is not None:
                # Read the image
                image = Image.open(webcam_image)
                st.session_state.input_image = np.array(image)
    
    # Process the image if available
    if st.session_state.input_image is not None:
        # Convert BGR to RGB if needed (OpenCV uses BGR)
        input_image = st.session_state.input_image
        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            input_image_cv = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        else:
            input_image_cv = input_image
        
        # Convert to sketch
        sketch = convert_to_sketch(input_image_cv, detail_level)
        
        # Store the sketch
        st.session_state.sketch_image = sketch
        
        # Display the sketch
        with col2:
            st.header("Sketch Output")
            st.image(sketch, caption=f"Sketch ({detail_text})", use_column_width=True)
            
            # Download button
            if st.session_state.sketch_image is not None:
                # Convert the sketch to bytes
                sketch_pil = Image.fromarray(st.session_state.sketch_image)
                buf = io.BytesIO()
                sketch_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Sketch",
                    data=byte_im,
                    file_name="sketchify_output.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main() 