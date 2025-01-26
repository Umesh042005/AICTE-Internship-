import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Constants
DEMO_IMAGE = 'IMG_20240104_163719.jpg'

# Body parts and  body pose pairs
BODY_PARTS = {
             "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
           "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                      "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Network dimensions of image 
inWidth, inHeight = 368, 368

# Load the pre-trained network
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Streamlit app
st.title("Human Pose Estimation using OpenCV")
st.text("Make sure you have a clear image with all the parts clearly visible.")

# Image uploader
img_file_buffer = st.file_uploader("Upload an image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

# Load image
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))

# Display original image
st.subheader("Original Image")
st.image(image, caption="Original Image", use_column_width=True)

# Threshold slider
thres = st.slider("Detection threshold for key points:", min_value=0, max_value=100, value=20, step=5) / 100

@st.cache_data
def poseDetector(frame: np.ndarray, threshold: float) -> np.ndarray:
    """Detect human poses in the given frame using OpenCV."""
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]

    # Preprocess input image
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    # Ensure BODY_PARTS matches the output size
    assert len(BODY_PARTS) == out.shape[1]

    # Detect points
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = int((frameWidth * point[0]) / out.shape[3])
        y = int((frameHeight * point[1]) / out.shape[2])
        points.append((x, y) if conf > threshold else None)

    # Draw skeleton
    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

# Process the image
output_image = poseDetector(image, thres)

# Display the output
st.subheader("Pose Estimation Output")
st.image(output_image, caption="Pose Estimation Output", use_column_width=True)

st.markdown("### Thank you for using this Human Pose Estimation tool!")

