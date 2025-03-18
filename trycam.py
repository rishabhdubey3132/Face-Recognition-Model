import cv2  
import os  
import numpy as np  

# Load the Haar Cascade classifier for face detection  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  

# Set paths for saving captured images and model  
data_folder = 'dataset'  
recognizer_path = 'trainer.yml'  

# Create the dataset folder if it does not exist  
if not os.path.exists(data_folder):  
    os.makedirs(data_folder)  

def capture_images(label):  
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend  
    if not cap.isOpened():  
        print("Error: Could not open camera. Please check your camera connection.")  
        return  

    # Set lower resolution  
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  

    count = 0  
    print(f"Capturing images for label: {label}. Press 'q' to stop capturing.")  

    while count < 30:  
        ret, frame = cap.read()  
        if not ret:  
            print("Failed to grab frame. Retrying...")  
            continue  # Skip to the next iteration if frame grab fails  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  

        for (x, y, w, h) in faces:  
            count += 1  
            face_img = gray[y:y+h, x:x+w]  
            img_path = os.path.join(data_folder, f"{label}.{count}.jpg")  
            cv2.imwrite(img_path, face_img)  
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  

        cv2.imshow('Capture Images', frame)  

        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  

    cap.release()  
    cv2.destroyAllWindows()  
    print("Image capturing completed. Total images captured:", count)  

def train_model():  
    print("Training model...")  
    images = []  
    labels = []  

    for filename in os.listdir(data_folder):  
        img_path = os.path.join(data_folder, filename)  
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale  

        # Extract the label from the filename, keeping it as string  
        label = filename.split('.')[0]  # Assuming label is the part before "."  
        
        # Only process images if they are valid  
        if image is not None:  
            images.append(image)  
            labels.append(label)  

    # Create a mapping from string labels to integer labels for training  
    unique_labels = list(set(labels))  
    label_map = {label: idx for idx, label in enumerate(unique_labels)}  
    
    # Convert the string labels to numerical labels for training  
    numeric_labels = [label_map[label] for label in labels]  

    if not images:  
        print("No images found to train the model.")  
        return  
    
    # Save the label mapping to use later for recognition  
    with open('label_mapping.txt', 'w') as f:  
        for label, idx in label_map.items():  
            f.write(f"{idx}:{label}\n")  

    # Train the LBPH face recognizer with numeric labels  
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()  
    face_recognizer.train(images, np.array(numeric_labels))  
    face_recognizer.save(recognizer_path)  

    print("Model trained and saved to", recognizer_path)  

def recognize_faces():  
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend  
    if not cap.isOpened():  
        print("Error: Could not open camera. Please check your camera connection.")  
        return  

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()  
    face_recognizer.read(recognizer_path)  

    # Read the label mapping  
    label_mapping = {}  
    with open('label_mapping.txt', 'r') as f:  
        for line in f:  
            idx, label = line.strip().split(':')  
            label_mapping[int(idx)] = label  

    print("Starting face recognition... Press 'q' to exit.")  

    while True:  
        ret, frame = cap.read()  
        if not ret:  
            print("Failed to grab frame. Exiting...")  
            break  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  

        for (x, y, w, h) in faces:  
            id_, confidence = face_recognizer.predict(gray[y:y+h, x:x+w])  
            if confidence < 100:  
                label_text = f"{label_mapping.get(id_, 'Unknown')}, Confidence: {round(100 - confidence, 2)}%"  
            else:  
                label_text = "Unknown"  

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  

        cv2.imshow('Face Recognition', frame)  

        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  

    cap.release()  
    cv2.destroyAllWindows()  
    print("Face recognition ended.")  

if __name__ == "__main__":  
    action = input("Choose action: (capture/train/recognize): ").strip().lower()  
    
    if action == 'capture':  
        label = input("Enter a label for the images: ")  
        capture_images(label)  
    elif action == 'train':  
        train_model()  
    elif action == 'recognize':  
        recognize_faces()  
    else:  
        print("Invalid action selected. Please choose 'capture', 'train', or 'recognize'.")