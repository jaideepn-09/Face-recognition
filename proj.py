import cv2 as cv
import numpy as np

def visualize(image, faces, thickness=2):
    if faces is None or faces[1] is None:
        return
    for idx, face in enumerate(faces[1]):
        coords = face[:-1].astype(np.int32)
        cv.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), thickness)
        cv.circle(image, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
        cv.circle(image, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
        cv.circle(image, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
        cv.circle(image, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
        cv.circle(image, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

def main():
    ref_image = cv.imread('reference_image.jpg')
    if ref_image is None:
        print("Error: Could not load reference image.")
        return

    faceDetector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (ref_image.shape[1], ref_image.shape[0]), 0.9, 0.3, 5000)
    faceInAdhaar = faceDetector.detect(ref_image)

    if faceInAdhaar is None or faceInAdhaar[1] is None:
        print("No faces detected in the reference image.")
        return

    recognizer = cv.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    print("Camera initialized successfully.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        faceDetector.setInputSize((frame.shape[1], frame.shape[0]))
        faceInQuery = faceDetector.detect(frame)

        if faceInQuery is not None and faceInQuery[1] is not None:
            visualize(frame, faceInQuery)
            
            face1_align = recognizer.alignCrop(ref_image, faceInAdhaar[1][0])
            
            for face_query in faceInQuery[1]:
                face2_align = recognizer.alignCrop(frame, face_query)
                
                face1_feature = recognizer.feature(face1_align)
                face2_feature = recognizer.feature(face2_align)

                cosine_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_COSINE)
                l2_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_NORM_L2)

                cosine_similarity_threshold = 0.363
                l2_similarity_threshold = 1.128

                msg_cosine = 'different identities'
                if cosine_score >= cosine_similarity_threshold:
                    msg_cosine = 'same identity'
                
                msg_l2 = 'different identities'
                if l2_score <= l2_similarity_threshold:
                    msg_l2 = 'same identity'

                coords = face_query[:-1].astype(np.int32)
                result_msg = f"Cosine: {msg_cosine}, L2: {msg_l2}"
                cv.putText(frame, result_msg, (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if msg_cosine == 'same identity' and msg_l2 == 'same identity':
                    cv.putText(frame, 'Authentication Confirmed!', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print("Authentication Confirmed!")
                    cv.imshow("Live Face Detection", frame)
                    cv.waitKey(5000)
                    cap.release()
                    cv.destroyAllWindows()
                    return

        cv.imshow("Live Face Detection", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
