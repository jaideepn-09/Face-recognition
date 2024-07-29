#Live Face Detection and Authentication

This project provides a real-time face detection and authentication system using OpenCV and pre-trained ONNX models. The application captures video from a camera, detects faces, and compares them to a reference image to verify identity. Once authentication is confirmed, the system automatically exits and displays a confirmation message.

Features
Real-Time Face Detection: Continuously detects faces from the camera feed.
Face Recognition: Compares detected faces with a reference image to authenticate identity.
Automatic Exit: Exits the video feed and displays a confirmation message once authentication is confirmed.
Live Display: Shows detected faces and similarity scores in real-time.
Requirements
Python 3.x
OpenCV (with ONNX support)
Pre-trained ONNX models for face detection and recognition
Setup
1. Install Dependencies
Ensure you have Python 3.x installed. Then, install the required Python packages:

bash
Copy code
pip install numpy opencv-python
2. Download ONNX Models
Download the following ONNX models and place them in the same directory as the script:

Face Detection Model: face_detection_yunet_2023mar.onnx
Face Recognition Model: face_recognition_sface_2021dec.onnx
You can obtain these models from the respective sources or repositories.

3. Prepare Your Images
Reference Image: Place an image file named reference_image.jpg in the same directory. This image will be used as the reference for authentication.
Query Image: The live camera feed will act as the query image.
Usage
Run the Script

Execute the script using Python:

bash
Copy code
python  proj.py -r REFERENCE_IMAGE -q QUERY_IMAGE
Replace proj.py with the name of your Python script file.

Authenticate

The script will open a camera feed and start detecting faces. Ensure that the reference image is visible in the camera feed or hold the reference image up to the camera. Once a face matches the reference image, the system will display "Authentication Confirmed!" and exit automatically.

Exit

You can also manually stop the script by pressing the q key while the camera feed window is active.

Code Explanation
visualize(image, faces, thickness=2): Annotates the image with detected faces and feature points.
main(): Handles loading images, initializing the camera, detecting faces, comparing identities, and managing the camera feed.
Troubleshooting
Camera Not Detected: Ensure the camera is properly connected and accessible. Test with other applications to confirm.
Face Detection Issues: Adjust the detection thresholds or verify that the models are correctly loaded.

Acknowledgements
OpenCV for the computer vision tools.
ONNX for the pre-trained models.
