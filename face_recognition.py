import cv2
import os
import numpy as np

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def capture_images(name, count=100):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capturing Images")
    img_counter = 0
    create_directory(f"dataset/{name}")

    while img_counter < count:
        ret, frame = cam.read()
        print(ret,frame)
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = gray[y:y+h, x:x+w]
            img_name = f"dataset/{name}/image_{img_counter}.jpg"
            cv2.imwrite(img_name, face)
            img_counter += 1
            color = (0, 255, 0)
            text_color = (255, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-30), (x+w, y), color, cv2.FILLED)
            cv2.putText(frame, f"{name} {img_counter}/100", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        cv2.imshow("Capturing Images", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    label_dict = {}

    for dir_name in dirs:
        if not os.path.isdir(os.path.join(data_folder_path, dir_name)):
            continue

        label = len(label_dict)
        label_dict[label] = dir_name
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue

            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces.append(image)
            labels.append(label)

    return faces, labels, label_dict

def train_face_recognizer(faces, labels):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    return face_recognizer

def recognize_faces(face_recognizer, label_dict):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face)

            if confidence < 100:
                name = label_dict[label]
                color = (0, 255, 0)
                text_color = (255, 255, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y-30), (x+w, y), color, cv2.FILLED)
                cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            else:
                color = (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y-30), (x+w, y), color, cv2.FILLED)
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Recognizing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def main():
    face_recognizer = None
    label_dict = None

    print(cv2.data.haarcascades)
    
    while True:
        print("Enter your option")
        choice = input("1. Train\n2. Recognize, q to Quit: ")
        if choice == '1':
            name = input("Enter the name of the person: ")
            capture_images(name)
            faces, labels, label_dict = prepare_training_data("dataset")
            face_recognizer = train_face_recognizer(faces, labels)
            print("Training completed.")
        elif choice == '2':
            if face_recognizer is not None and label_dict is not None:
                recognize_faces(face_recognizer, label_dict)
            else:
                print("No trained data found. Please train first.")
        elif choice == 'q':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
