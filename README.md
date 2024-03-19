# Face_Recognition_System
K Nearest Neighbour classification algorithm in face recognition using openCV and HaarCascades CNN Model for frontal face detection.

To run the face recognition program, follow these steps:

1. **Install Required Libraries:**
   Ensure you have the required libraries installed. You can install them using pip if you haven't already:
   ```bash
   pip install opencv-python scikit-learn
   ```

2. **Download the CelebA Dataset:**
   Download the CelebA dataset from its official website: [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Extract the dataset to a folder on your computer.

3. **Modify the Code:**
   Open the code in a text editor or Python IDE. Replace `'path/to/celeba_dataset'` in the code with the actual path where you extracted the CelebA dataset.

4. **Run the Python Script:**
   Save the modified code as a Python file, for example, `face_recognition.py`. Open a terminal or command prompt, navigate to the directory containing the script, and run it using the following command:
   ```bash
   python face_recognition.py
   ```

5. **Use the Webcam:**
   Once you run the script, it will open your webcam (if available) and start detecting faces in real-time. You should see rectangles drawn around detected faces, along with their labels predicted by the KNN classifier.

6. **Quit the Program:**
   To exit the program, press the 'q' key on your keyboard.

Make sure your computer has a webcam connected and configured properly for this program to work. Also, ensure that the OpenCV library can access your webcam. If you encounter any errors, check the console output for error messages and troubleshoot accordingly.
