import cv2  
import os  
import numpy as np  
import pyttsx3  # For text-to-speech  
import time  # For introducing delays  

# Initialize text-to-speech engine  
engine = pyttsx3.init()  

def speak(message):  
    engine.say(message)  
    engine.runAndWait()  

# Load the Haar Cascade classifier for face detection  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  

# Set paths for saving captured images and model  
data_folder = 'dataset'  
recognizer_path = 'trainer.yml'  
label_mapping_path = 'label_mapping.txt'  

# Create the dataset folder if it does not exist  
if not os.path.exists(data_folder):  
    os.makedirs(data_folder)  

def is_image_sharp(image, threshold=40):  
    # Calculate the Laplacian variance for sharpness  
    laplacian = cv2.Laplacian(image, cv2.CV_64F)  
    variance = laplacian.var()  
    return variance >= threshold  

def capture_images(label):  
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
    if not cap.isOpened():  
        print("Error: Could not open camera. Please check your camera connection.")  
        return  

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  

    count = 0  
    print(f"Capturing images for label: {label}. Press 'q' to stop capturing.")  
    speak("Please move your face left, right, up, or down to capture images.")  

    while count < 40:  # Capturing a maximum of 40 images
        ret, frame = cap.read()  
        if not ret:  
            print("Failed to grab frame. Retrying...")  
            continue  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  

        if len(faces) == 0:  
            continue  

        for (x, y, w, h) in faces:  
            face_img = gray[y:y+h, x:x+w]  
            
            # Check if the image is sharp enough
            if not is_image_sharp(face_img, threshold=40):  
                print("Image is too blurry, skipping.")  
                continue  

            img_path = os.path.join(data_folder, f"{label}.{count + 1}.jpg")  
            cv2.imwrite(img_path, face_img)  
            print(f"Captured image {count + 1}")  
            count += 1  

            # Add a 0.1 second delay after capturing each image  
            time.sleep(0.15)  

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

    # Load images and labels from dataset  
    for filename in os.listdir(data_folder):  
        img_path = os.path.join(data_folder, filename)  
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  

        # Extract the label from the filename  
        label = filename.split('.')[0]  

        if image is not None:  
            images.append(image)  
            labels.append(label)  

    # Create a mapping from string labels to integer labels  
    unique_labels = list(set(labels))  
    label_map = {label: idx for idx, label in enumerate(unique_labels)}  
    numeric_labels = [label_map[label] for label in labels]  

    if not images:  
        print("No images found to train the model.")  
        return  

    # Save the label mapping  
    with open(label_mapping_path, 'w') as f:  
        for label, idx in label_map.items():  
            f.write(f"{idx}:{label}\n")  

    # Train the LBPH face recognizer  
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()  
    face_recognizer.train(images, np.array(numeric_labels))  
    face_recognizer.save(recognizer_path)  

    print("Model trained and saved to", recognizer_path)  

def recognize_faces():  
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
    if not cap.isOpened():  
        print("Error: Could not open camera. Please check your camera connection.")  
        return  

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  

    # Load the trained model  
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()  
    face_recognizer.read(recognizer_path)  

    # Read the label mapping  
    label_mapping = {}  
    with open(label_mapping_path, 'r') as f:  
        for line in f:  
            idx, label = line.strip().split(':')  
            label_mapping[int(idx)] = label  

    greeted_users = set()  
    last_label_text = ""  # Track the last spoken message  
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

            # Determine the message based on confidence levels  
            if confidence < 20:  
                label_text = "I'm sorry, I don't recognize you."  
            elif confidence < 40:  
                label_text = f"I think you're {label_mapping.get(id_, 'Unknown')}."  
            elif confidence < 60:  
                label_text = f"Oh, I know you. You are {label_mapping.get(id_, 'Unknown')}."  
            elif confidence <= 80:  
                label_text = f"Oh, I'm so sure you're {label_mapping.get(id_, 'Unknown')}."  
            else:  
                label_text = "Unknown"  

            # Only speak if the label text has changed and confidence has gone up  
            if id_ not in greeted_users and (last_label_text != label_text):  
                last_label_text = label_text  
                greeted_users.add(id_)  
                time.sleep(4)  # Delay for smoother interaction  
                speak(label_text)  

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  

        cv2.imshow('Face Recognition', frame)  

        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  

    cap.release()  
    cv2.destroyAllWindows()  
    print("Face recognition ended.")  

def delete_dataset():  
    print("Deleting dataset and trained model...")  
    # Delete images in the dataset folder  
    for filename in os.listdir(data_folder):  
        file_path = os.path.join(data_folder, filename)  
        try:  
            if os.path.isfile(file_path):  
                os.unlink(file_path)  
        except Exception as e:  
            print(f"Error deleting file {file_path}: {e}")  

    # Delete the trainer.yml model file  
    if os.path.isfile(recognizer_path):  
        try:  
            os.unlink(recognizer_path)  
            print(f"Deleted the trained model: {recognizer_path}")  
        except Exception as e:  
            print(f"Error deleting trained model {recognizer_path}: {e}")  

    print("Dataset and trained model deleted.")  

def clear_data():  
    print("Clearing label mapping and trained model...")  
    try:  
        if os.path.isfile(label_mapping_path):  
            os.unlink(label_mapping_path)  
        if os.path.isfile(recognizer_path):  
            os.unlink(recognizer_path)  
    except Exception as e:  
        print(f"Error clearing data: {e}")  
    print("Label mapping and trained model cleared.")  

def main():  
    while True:  
        action = input("Choose action: (capture/train/recognize/delete/exit): ").strip().lower()  

        if action == 'capture':  
            label = input("Enter a label for the images: ")  
            capture_images(label)  
        elif action == 'train':  
            train_model()  
        elif action == 'recognize':  
            recognize_faces()  
        elif action == 'delete':  
            delete_dataset()  
        elif action == 'exit':  
            print("Exiting program.")  
            break  
        else:  
            print("Invalid action selected. Please choose 'capture', 'train', 'recognize', 'delete', or 'exit'.")  

if __name__ == "__main__":  
    main()
