# Sign Language Interpreter

A real-time sign language recognition system that uses a webcam to detect hand gestures, classify them into letters using a trained CNN, and speak out the resulting words/sentences aloud.

## How It Works

The pipeline has three stages:

1. **Data Collection** (`data_collection.py`) — Captures webcam frames, detects the hand using MediaPipe/cvzone, extracts a normalized "skeleton" image (hand landmarks drawn on a plain white background), and saves it as training data for a chosen letter label.

2. **Training** (`train_model.py`) — Trains a Convolutional Neural Network (CNN) on the collected skeleton images to classify hand shapes into letters (A-Z).

3. **Inference** (`main_inference.py`) — Runs live webcam detection, classifies each frame in real time, assembles detected letters into words and sentences, and uses text-to-speech to read the result aloud once you pause.

## Why Skeleton Images?

Rather than training directly on raw webcam frames (which vary a lot in lighting, skin tone, and background), the pipeline extracts just the hand's landmark skeleton and draws it on a plain white canvas. This keeps the model focused on hand *shape* rather than incidental visual noise, and makes training more consistent across sessions.

## Tech Stack

- **OpenCV** — webcam capture and image processing
- **MediaPipe / cvzone** — hand landmark detection
- **TensorFlow / Keras** — CNN model training and inference
- **pyttsx3** — offline text-to-speech
- **keyboard** — global keypress detection for on-screen controls

## Project Structure

```
├── data_collection.py    # Collect labeled hand-sign training images
├── train_model.py        # Train the CNN classifier
├── main_inference.py     # Run live sign-to-speech recognition
├── labels.txt            # Class labels (auto-generated after training)
├── requirements.txt      # Python dependencies
└── Data2/                # Collected training images (not included in repo)
```

> **Note:** The trained model file (`sign_language_model.h5`) and the collected training data (`Data2/`) are not included in this repository due to file size. See "Running the Project" below to regenerate them.

## Setup

**1. Create a virtual environment (Python 3.11 recommended):**
```bash
python -m venv signenv
signenv\Scripts\activate      # Windows
source signenv/bin/activate   # macOS/Linux
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Project

**Step 1 — Collect training data** (run once per letter you want to recognize):
```bash
python data_collection.py
```
You'll be prompted for a letter label. Hold your hand in frame and press **s** to save a sample image, **q** to stop collecting for that letter. Aim for 150-200+ images per letter, varying hand angle/position slightly for better generalization.

**Step 2 — Train the model:**
```bash
python train_model.py
```
This trains a CNN on everything in `Data2/` and saves `sign_language_model.h5` + `labels.txt`. Training uses data augmentation and early stopping to reduce overfitting.

**Step 3 — Run live inference:**
```bash
python main_inference.py
```
Hold a sign steady in frame to register a letter. Controls (work anywhere on your desktop, not just the video window):
- **Space** — manually insert a word break
- **Enter** — speak the current sentence aloud immediately
- **Backspace** — undo the last character/word
- **Q** — quit

The system also auto-speaks the current sentence after a period of inactivity, and auto-inserts word breaks when your hand leaves the frame for a couple of seconds.

## Known Limitations

- Accuracy depends heavily on the quantity and variety of training data collected — more samples per letter, captured across different lighting/angles, generalize better.
- Visually similar hand shapes (e.g. certain letter pairs in fingerspelling) can be harder for the model to distinguish.
- Currently supports single-hand detection only.

## Possible Future Improvements

- Expand to word/phrase-level gesture recognition instead of letter-by-letter spelling
- Add a larger, more diverse training dataset across multiple people
- Support two-handed signs
- Package as a standalone desktop app
