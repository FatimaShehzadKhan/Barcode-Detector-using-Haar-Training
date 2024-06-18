
import cv2

print("Hello, world!")

# Wait for user input to prevent the window from closing
input("Press Enter to exit...")

# Load the pre-trained XML file
barcode_cascade = cv2.CascadeClassifier('barcode (1).xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Detect barcodes in the frame
    barcodes = barcode_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through the detected barcodes
    for (x, y, w, h) in barcodes:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with the detected barcodes
    cv2.imshow('Object Detection', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the window
input()

cap.release()
cv2.destroyAllWindows()

