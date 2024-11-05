import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle
import os
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Initialize the FaceAnalysis app
app_insight = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app_insight.prepare(ctx_id=0, det_size=(640, 640))

# Specify the new database file name
database_file = "embedding_database_arcface.pkl"

# Load known embeddings from the new database file or create a new dictionary
try:
    with open(database_file, "rb") as f:
        embedding_database = pickle.load(f)
        if not isinstance(embedding_database, dict):
            embedding_database = {}
        else:
            # Normalize stored embeddings
            for name in embedding_database:
                embeddings = embedding_database[name]
                if isinstance(embeddings, np.ndarray):
                    embeddings = [embeddings]
                normalized_embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
                embedding_database[name] = normalized_embeddings
except FileNotFoundError:
    embedding_database = {}
    # Create a new database file if it doesn't exist
    with open(database_file, "wb") as f:
        pickle.dump(embedding_database, f)

# Function to recognize the face
def recognize_face(new_embedding, embedding_database, threshold=0.4):  # Adjusted threshold
    if len(embedding_database) == 0:
        return "Unknown", 0.0

    max_similarity = -1
    identity = "Unknown"

    # Normalize the new embedding
    new_embedding_norm = new_embedding / np.linalg.norm(new_embedding)

    for name, embeddings in embedding_database.items():
        for idx, stored_embedding in enumerate(embeddings):
            # Compute cosine similarity
            similarity = np.dot(stored_embedding, new_embedding_norm)
            if similarity > threshold and similarity > max_similarity:
                max_similarity = similarity
                identity = name

    if identity == "Unknown":
        max_similarity = 0.0  # Set similarity to 0 if unknown

    return identity, max_similarity

# Function to display image with detected or recognized names
def display_image_with_name(img, window_title="Image"):
    # Convert image from BGR to RGB for displaying
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Create a new window to display the image
    top = ctk.CTkToplevel()
    top.title(window_title)
    top.geometry("800x600")

    # Bring the window to the front
    top.lift()
    top.attributes('-topmost', True)
    top.after_idle(top.attributes, '-topmost', False)
    top.focus_force()

    img_label = ctk.CTkLabel(top, text="")
    img_label.pack(pady=10)

    img_display = pil_img.copy()
    img_display.thumbnail((800, 600))
    img_tk = ImageTk.PhotoImage(img_display)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

# Function to add a new face to the embedding database
def add_new_face(image_path, name):
    # Load the image from the given path
    img = cv2.imread(image_path)
    if img is None:
        messagebox.showerror("Error", f"Unable to load image at {image_path}")
        return

    # Detect faces and get embeddings using InsightFace
    faces = app_insight.get(img)

    if faces:
        face_added = False
        for face in faces:
            # Get the face embedding and normalize it
            embedding = face.embedding
            embedding_norm = embedding / np.linalg.norm(embedding)

            # Add the new face embedding to the database
            if name not in embedding_database:
                embedding_database[name] = []
            embedding_database[name].append(embedding_norm)
            face_added = True

            # Draw the bounding box and name on the image
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if face_added:
            with open(database_file, "wb") as f:
                pickle.dump(embedding_database, f)
            messagebox.showinfo("Success", f"Face(s) enrolled as '{name}'")

            # Display the image with name(s)
            display_image_with_name(img, f"Enrolled: {name}")
        else:
            messagebox.showerror("Error", "No face detected in the provided image.")
    else:
        messagebox.showerror("Error", "No face detected in the provided image.")

# Function to add a new face using the webcam
def add_new_face_from_camera(name):
    # Use the webcam to capture an image
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera")
        return
    ret, img = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "Unable to capture image from camera")
        return

    # Detect faces and get embeddings using InsightFace
    faces = app_insight.get(img)

    if faces:
        face_added = False
        for face in faces:
            # Get the face embedding and normalize it
            embedding = face.embedding
            embedding_norm = embedding / np.linalg.norm(embedding)

            # Add the new face embedding to the database
            if name not in embedding_database:
                embedding_database[name] = []
            embedding_database[name].append(embedding_norm)
            face_added = True

            # Draw the bounding box and name on the image
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if face_added:
            with open(database_file, "wb") as f:
                pickle.dump(embedding_database, f)
            messagebox.showinfo("Success", f"Face(s) enrolled as '{name}'")

            # Display the image with name(s)
            display_image_with_name(img, f"Enrolled: {name}")
        else:
            messagebox.showerror("Error", "No face detected in the captured image.")
    else:
        messagebox.showerror("Error", "No face detected in the captured image.")

# Function to open a file dialog and select an image
def select_image():
    file_path = filedialog.askopenfilename()
    return file_path

# Function to handle adding a new face
def handle_add_face():
    name_window = ctk.CTkToplevel()
    name_window.title("Enter Name")
    name_window.geometry("300x150")

    # Bring the window to the front
    name_window.lift()
    name_window.attributes('-topmost', True)
    name_window.after_idle(name_window.attributes, '-topmost', False)
    name_window.focus_force()

    ctk.CTkLabel(name_window, text="Enter Name:", font=ctk.CTkFont(size=12)).pack(pady=10)
    name_entry = ctk.CTkEntry(name_window, font=ctk.CTkFont(size=12))
    name_entry.pack(pady=5)

    def handle_name():
        name = name_entry.get().strip()
        if name:
            response = messagebox.askquestion("Add Face", "Do you want to add a face from an image file? Click 'No' to use the camera.")
            if response == 'yes':
                image_path = select_image()
                if image_path:
                    add_new_face(image_path, name)
            else:
                add_new_face_from_camera(name)
            name_window.destroy()
        else:
            messagebox.showerror("Error", "Please enter a name for the person.")

    ctk.CTkButton(name_window, text="Submit", command=handle_name, font=ctk.CTkFont(size=12)).pack(pady=10)

# Function to detect and recognize faces
def handle_recognize_face():
    if hasattr(handle_recognize_face, 'option_window') and handle_recognize_face.option_window.winfo_exists():
        # Window already open
        return
    option_window = ctk.CTkToplevel()
    handle_recognize_face.option_window = option_window
    option_window.title("Select Recognition Option")
    option_window.geometry("400x200")

    # Bring the window to the front
    option_window.lift()
    option_window.attributes('-topmost', True)
    option_window.after_idle(option_window.attributes, '-topmost', False)
    option_window.focus_force()

    ctk.CTkLabel(option_window, text="Select Recognition Option:", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=20)

    def select_image():
        option_window.destroy()
        image_path = filedialog.askopenfilename()
        if image_path:
            detect_and_recognize_face(image_path=image_path)

    def select_video():
        option_window.destroy()
        video_path = filedialog.askopenfilename()
        if video_path:
            detect_and_recognize_face(video_path=video_path)

    def use_camera():
        option_window.destroy()
        # Present options to take image or record video
        camera_option_window = ctk.CTkToplevel()
        camera_option_window.title("Camera Option")
        camera_option_window.geometry("400x200")

        # Bring the window to the front
        camera_option_window.lift()
        camera_option_window.attributes('-topmost', True)
        camera_option_window.after_idle(camera_option_window.attributes, '-topmost', False)
        camera_option_window.focus_force()

        ctk.CTkLabel(camera_option_window, text="Select Camera Option:", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=20)

        def take_image():
            camera_option_window.destroy()
            detect_and_recognize_face(use_camera_image=True)

        def record_video():
            camera_option_window.destroy()
            detect_and_recognize_face(use_camera_video=True)

        ctk.CTkButton(camera_option_window, text="Take Image", command=take_image, font=ctk.CTkFont(size=12)).pack(pady=5)
        ctk.CTkButton(camera_option_window, text="Record Video", command=record_video, font=ctk.CTkFont(size=12)).pack(pady=5)

    ctk.CTkButton(option_window, text="Image File", command=select_image, font=ctk.CTkFont(size=12)).pack(pady=5)
    ctk.CTkButton(option_window, text="Video File", command=select_video, font=ctk.CTkFont(size=12)).pack(pady=5)
    ctk.CTkButton(option_window, text="Use Camera", command=use_camera, font=ctk.CTkFont(size=12)).pack(pady=5)

# Function to process a frame
def process_frame(frame):
    # Detect faces
    faces = app_insight.get(frame)
    if faces:
        for face in faces:
            # Get the face embedding and normalize it
            embedding = face.embedding
            embedding_norm = embedding / np.linalg.norm(embedding)

            # Recognize the face
            name, similarity = recognize_face(embedding_norm, embedding_database)
            similarity_percentage = similarity * 100  # Convert to percentage

            # Prepare label with name and similarity
            label = f"{name}: {similarity_percentage:.2f}%"

            # Draw bounding box and name on frame
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

# Function to detect and recognize faces
def detect_and_recognize_face(image_path=None, video_path=None, use_camera_image=False, use_camera_video=False):
    if image_path:
        # Load the image from the given path
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("Error", f"Unable to load image at {image_path}")
            return

        # Process image
        img = process_frame(img)

        # Display the image
        display_image_with_name(img, "Recognition Results")

    elif video_path:
        # Process video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Unable to open video file {video_path}")
            return

        # Get the width and height of the frames
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get FPS of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None or np.isnan(fps):
            fps = 15  # Set a default FPS value
        else:
            fps = int(fps)

        # Define the codec and create VideoWriter object for MP4
        output_file = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Process frame
            frame = process_frame(frame)

            # Write frame to output video file
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        messagebox.showinfo("Processing Completed", f"Processed video saved to {output_file}")

    elif use_camera_image:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to access the camera")
            return
        ret, img = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error", "Unable to capture image from camera")
            return

        # Process image
        img = process_frame(img)

        # Display the image
        display_image_with_name(img, "Recognition Results")

    elif use_camera_video:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to access the camera")
            return

        # Set the desired FPS for VideoWriter
        fps = 10  # Adjust as needed

        # Get the width and height of the frames
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object for MP4
        output_file = 'output_camera_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

        # Flag to control recording
        recording = [True]  # Using list to make it mutable in nested function

        # Function to process video frames
        def record_video():
            while recording[0]:
                ret, frame = cap.read()
                if not ret:
                    break  # Unable to read frame

                # Process frame
                frame = process_frame(frame)

                # Write frame to output video file
                out.write(frame)

            cap.release()
            out.release()
            cv2.destroyAllWindows()

        # Function to stop recording
        def stop_recording():
            recording[0] = False
            record_window.destroy()
            messagebox.showinfo("Recording Stopped", f"Recording stopped. Output video saved to {output_file}")

        # Create a window with "Stop Recording" button
        record_window = ctk.CTkToplevel()
        record_window.title("Recording Video")
        record_window.geometry("300x100")

        # Bring the window to the front
        record_window.lift()
        record_window.attributes('-topmost', True)
        record_window.after_idle(record_window.attributes, '-topmost', False)
        record_window.focus_force()

        ctk.CTkLabel(record_window, text="Recording...").pack(pady=10)
        ctk.CTkButton(record_window, text="Stop Recording", command=stop_recording).pack(pady=10)

        # Start recording in a separate thread
        threading.Thread(target=record_video).start()

# Create the GUI application
ctk.set_appearance_mode("dark")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (default), "green", "dark-blue"

app = ctk.CTk()
app.title("Face Recognition System")
app.geometry("600x400")

# Bring the main window to the front
app.lift()
app.attributes('-topmost', True)
app.after_idle(app.attributes, '-topmost', False)
app.focus_force()

# Header Frame for Modern Look
header_frame = ctk.CTkFrame(app, height=80)
header_frame.pack(fill="x")

header_label = ctk.CTkLabel(header_frame, text="Face Recognition System", font=ctk.CTkFont(size=20, weight="bold"))
header_label.pack(pady=20)

# UI Elements
frame = ctk.CTkFrame(app)
frame.pack(pady=20, padx=20, fill="both", expand=True)

# Centering the buttons using pack()
button1 = ctk.CTkButton(frame, text="Add New Face", command=handle_add_face, font=ctk.CTkFont(size=14, weight="bold"))
button1.pack(pady=10, anchor='center')

button2 = ctk.CTkButton(frame, text="Recognize Face", command=handle_recognize_face, font=ctk.CTkFont(size=14, weight="bold"))
button2.pack(pady=10, anchor='center')

# Run the application
app.mainloop()
