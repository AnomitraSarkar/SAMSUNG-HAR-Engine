# SAMSUNG-HAR-Engine
Here’s a properly formatted **README.md** file:  

```markdown
# UCF50 Action Recognition Project

This project uses the UCF50 dataset for action recognition, implemented with TensorFlow and other Python libraries. It provides a script (`main.py`) to perform action classification using deep learning models.

```
---

## How to Run `main.py`

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AnomitraSarkar/SAMSUNG-HAR-Engine.git
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd SAMSUNG-HAR-Engine
   ```

3. **Install the Required Packages**:
   Ensure you have Python installed. Then, install the necessary packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Script**:
   Execute the `main.py` script:
   ```bash
   python main.py
   ```

---

## Action List Based on UCF50

The UCF50 dataset comprises the following 50 action categories:

- BaseballPitch  
- Basketball  
- BenchPress  
- Biking  
- Billiards  
- BreastStroke  
- CleanAndJerk  
- Diving  
- Drumming  
- Fencing  
- GolfSwing  
- HighJump  
- HorseRace  
- HorseRiding  
- HulaHoop  
- JavelinThrow  
- JugglingBalls  
- JumpRope  
- JumpingJack  
- Kayaking  
- Lunges  
- MilitaryParade  
- Mixing  
- NunChucks  
- PizzaTossing  
- PlayingGuitar  
- PlayingPiano  
- PlayingViolin  
- PoleVault  
- PommelHorse  
- PullUps  
- Punch  
- PushUps  
- RockClimbingIndoor  
- RopeClimbing  
- Rowing  
- SalsaSpin  
- SkateBoarding  
- Skiing  
- Skijet  
- SoccerJuggling  
- Surfing  
- Swing  
- TableTennisShot  
- TaiChi  
- TennisSwing  
- ThrowDiscus  
- TrampolineJumping  
- VolleyballSpiking  
- WalkingWithDog  

*Note: This list is directly derived from the UCF50 dataset.*

## 5 Other actions added in UCF50 dataset contain the following action, as follows

- Clapping
- Running
- Turning around
- Walking
- Waving

---

## Installing Required Packages

The project requires the following Python packages:

- `os` (Standard Library)  
- `cv2` (OpenCV)  
- `csv` (Standard Library)  
- `pafy`  
- `math` (Standard Library)  
- `random` (Standard Library)  
- `numpy`  
- `datetime` (Standard Library)  
- `tensorflow`  
- `collections` (Standard Library)  
- `matplotlib`  
- `moviepy`  
- `sklearn`  
- `customtkinter`  
- `tkinter` (Standard Library)  

To install the required packages, use the following command:
```bash
pip install opencv-python pafy numpy tensorflow matplotlib moviepy scikit-learn customtkinter
```

*Note: Standard library modules (`os`, `csv`, `math`, `random`, `datetime`, `collections`, `tkinter`) are included with Python and do not require separate installation.*

---

## Additional Notes

- Ensure your Python environment is correctly set up and all dependencies are installed.
- If you encounter any issues, please refer to the project's documentation or open an issue in the repository.

---

## Acknowledgments

This project is based on the UCF50 dataset for action recognition tasks. Proper attribution to the dataset's original creators is included.

Anomitra Sarkar - Codeups
Achuth G Model - Recommendations
Chaitanya Patil - Dataset
