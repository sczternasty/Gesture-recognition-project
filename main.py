import mediapipe as mp
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
model_path = 'gesture_recognizer.task'
base_options = BaseOptions(model_asset_path=model_path)
class Mediapipe_GestureModule():
    def __init__(self):
        self.mp_drawing = solutions.drawing_utils
        self.mp_pose = solutions.pose
        self.results = None

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        # pose_landmarks_list = detection_result
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

    # Create a pose landmarker instance with the live stream mode:
    def print_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        print('gesture recognition result: {}'.format(result))

    def main(self):
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result)

        video = cv2.VideoCapture(0)

        timestamp = 0
        with GestureRecognizer.create_from_options(options) as recognizer:

            while video.isOpened():
                # Capture frame-by-frame
                ret, frame = video.read()

                if not ret:
                    print("Ignoring empty frame")
                    break

                timestamp += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                self.results = recognizer.recognize_async(mp_image, timestamp)



                if (not (self.results is None)):
                    annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), self.results)
                    # cv2.imshow('Show',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                    cv2.imshow('Show', annotated_image)
                    print("showing detected image")
                else:
                    cv2.imshow('Show', frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    print("Closing Camera Stream")
                    break

            video.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    body_module = Mediapipe_GestureModule()
    body_module.main()