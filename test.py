import cv2
import threading
import time

""" THIS WORKS HOLY FUICKING SHIT"""
class WebcamChecker:
    def __init__(self, update_interval=5):
        self.update_interval = update_interval  # Time in seconds to wait between checks
        self.webcams = []
        self.running = False

    def list_connected_webcams(self):
        while self.running:
            new_webcams = []
            for i in range(10):  # You can increase this range if necessar
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    new_webcams.append((i, f"Webcam {i}"))
                    cap.release()
            self.webcams = new_webcams
            self.display_webcams()  # Display updated list
            time.sleep(self.update_interval)  # Wait before checking again

    def display_webcams(self):
        if self.webcams:
            print("Connected Webcams:")
            for index, name in self.webcams:
                print(f"Device Index: {index}, Name: {name}")
        else:
            print("No webcams found.")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.list_connected_webcams)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()  # Wait for the thread to finish


# Example usage
if __name__ == "__main__":
    webcam_checker = WebcamChecker(update_interval=5)  # Check every 5 seconds
    try:
        webcam_checker.start()
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        webcam_checker.stop()
        print("Webcam checker stopped.")
