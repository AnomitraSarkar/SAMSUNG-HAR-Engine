import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import csv
import tensorflow as tf
import cv2
import numpy as np
from collections import deque
import datetime as dt
from moviepy import VideoFileClip


# Initialize CustomTkinter
ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "dark-blue", "green"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Video Processing Tool")
        self.geometry("500x500")
        
        # Variables to hold file paths
        self.model_path = ctk.StringVar()
        self.video_path = ctk.StringVar()
        self.output_dir = ctk.StringVar()

        # Input Model Selection
        self.model_label = ctk.CTkLabel(self, text="Select Input Model:")
        self.model_label.pack(pady=(20, 5))
        self.model_entry = ctk.CTkEntry(self, textvariable=self.model_path, placeholder_text="Choose model file")
        self.model_entry.pack(pady=5, fill="x", padx=20)
        self.model_button = ctk.CTkButton(self, text="Browse", command=self.browse_model)
        self.model_button.pack(pady=5)

        # Input Video Selection
        self.video_label = ctk.CTkLabel(self, text="Select Input Video:")
        self.video_label.pack(pady=(20, 5))
        self.video_entry = ctk.CTkEntry(self, textvariable=self.video_path, placeholder_text="Choose video file")
        self.video_entry.pack(pady=5, fill="x", padx=20)
        self.video_button = ctk.CTkButton(self, text="Browse", command=self.browse_video)
        self.video_button.pack(pady=5)

        # Output Directory Selection
        self.output_label = ctk.CTkLabel(self, text="Select Output Directory:")
        self.output_label.pack(pady=(20, 5))
        self.output_entry = ctk.CTkEntry(self, textvariable=self.output_dir, placeholder_text="Choose output directory")
        self.output_entry.pack(pady=5, fill="x", padx=20)
        self.output_button = ctk.CTkButton(self, text="Browse", command=self.browse_output_dir)
        self.output_button.pack(pady=5)

        # Process Button
        self.process_button = ctk.CTkButton(self, text="Process", command=self.process_files)
        self.process_button.pack(pady=(30, 10))

    def browse_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.h5;*.onnx;*.pt"), ("All Files", "*.*")])
        if file_path:
            self.model_path.set(file_path)

    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv"), ("All Files", "*.*")])
        if file_path:
            self.video_path.set(file_path)

    def browse_output_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir.set(dir_path)

    def process_files(self):
        model_file = self.model_path.get()
        video_file = self.video_path.get()
        output_directory = self.output_dir.get()

        if not model_file or not video_file or not output_directory:
            messagebox.showerror("Error", "Please select the model file, video file, and output directory.")
            return

        if not os.path.exists(model_file) or not os.path.exists(video_file):
            messagebox.showerror("Error", "One or both file paths are invalid.")
            return

        if not os.path.exists(output_directory):
            messagebox.showerror("Error", "Output directory does not exist.")
            return

        # try:
                       
        print(model_file, video_file, output_directory)
        
        process_video(model_file, video_file, output_directory)

        messagebox.showinfo("Success", f"Files processed successfully!\n\nSaved to:\n{output_directory}")
        # except Exception as e:
        #     messagebox.showerror("Processing Error", f"An error occurred: {str(e)}")

def process_video(model_file, video_file, output_directory):
    global CLASSES_LIST, IMAGE_HEIGHT, IMAGE_WIDTH
    CLASSES_LIST = ['BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards', 'BreastStroke', 'CleanAndJerk', 'Diving', 'Drumming', 'Fencing', 'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Lunges', 'MilitaryParade', 'Mixing', 'Nunchucks', 'PizzaTossing', 'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Swing', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog', 'YoYo']
    SEQUENCE_LENGTH = 20
    IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
    model = tf.keras.models.load_model(filepath=model_file)
    input_video_filepath = video_file
    output_video_filepath = f"{output_directory}/prediction_video-output.mp4"
    result_csv_filepath = f"{output_directory}/prediction_csv-output.csv"
    predict_on_video(input_video_filepath, output_video_filepath, SEQUENCE_LENGTH, model, result_csv_filepath)
    VideoFileClip(output_video_filepath, audio=False, target_resolution=(300,None)).preview()
    
def append_on_csv(filename, dataline, mode="a"):
    with open(filename, mode,newline='\n') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(dataline) 
        
def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH, model, result_csv_filepath):
    current_date_time_start = dt.datetime.now()
    '''
    This function will perform action recognition on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    output_file_path: The path where the ouput video with the predicted action being performed overlayed will be stored.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter Object to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                               video_reader.get(cv2.CAP_PROP_FPS), 
                               (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():

        # Read the frame.
        ok, frame = video_reader.read() 
        
        # Check if frame is not read properly then break the loop.
        if not ok:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255
        confidence = 0
        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:

            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis = 0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)
            confidence = predicted_labels_probabilities[predicted_label]
            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]
            current_date_time_iter = dt.datetime.now()
            delta = current_date_time_iter - current_date_time_start
            append_on_csv(result_csv_filepath,[100*confidence,predicted_class_name,f"{int(delta.total_seconds()//60)}hr {delta.total_seconds()%60}s"]) #will use current_date_time_string cuz i can't figure it out
            print(f"{100*confidence}% chance of being {predicted_class_name} {int(delta.total_seconds()//60)}min {delta.total_seconds()%60}s") #will use current_date_time_string cuz i can't figure it out

        # Write predicted class name on top of the frame.
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write The frame into the disk using the VideoWriter Object.
        video_writer.write(frame)
        
    # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    video_writer.release()   

if __name__ == "__main__":
    app = App()
    app.mainloop()
