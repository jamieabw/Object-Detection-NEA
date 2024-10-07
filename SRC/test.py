import subprocess
import cv2

def list_webcams_with_names():
    # Use wmic to get information about connected video capture devices (Windows-specific)
    try:
        result = subprocess.run(['wmic', 'path', 'Win32_PnPEntity', 'where', 'Caption like "%cam%"', 'get', 'Caption'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8').strip().split("\n")[1:]  # Skip the header line

        # Clean up the output and remove any empty lines
        device_names = [line.strip() for line in output if line.strip()]
    
        print("Available webcams:")
        webcams = {}
        
        # Loop through possible OpenCV sources (device IDs 0-9)
        for idx, name in enumerate(device_names):
            print(f"Webcam source ID: {idx}, Name: {name}")
            webcams.update({idx : name})
            # Check if OpenCV can access the device
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"OpenCV can access webcam {idx}: {name}")
                cap.release()
            else:
                print(f"OpenCV cannot access webcam {idx}: {name}")

        print(webcams)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_webcams_with_names()
