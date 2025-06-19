# Wakeword Detection Android App

This Android application performs real-time wake word detection using ONNX models. The system integrates custom audio preprocessing, Mel-spectrogram generation, feature embedding, and wake word classification — all directly on-device. The backend is fully implemented in Java, optimized for Android runtime environments.

## Features

- **Real-Time Wake Word Detection**: Detects custom wake words from live microphone input.
- **ONNX Runtime Integration**: Runs ONNX-based inference for Mel-spectrogram, embedding, and classification models.
- **Custom Pipeline**: Uses a modular Java pipeline: audio buffer → Mel-spectrogram → embedding → wake word prediction.
- **YAMNet Classification**: Optionally integrates TensorFlow Lite YAMNet to classify ambient audio.
- **Thread-Safe Design**: Handles audio recording, inference, and UI updates with optimized multithreaded architecture.
- **Modular & Extensible**: Easily replace models or adjust buffer/window configurations for different use cases.

## Architecture Overview

```text
AudioRecord (16 kHz PCM)
       ↓
1280-sample sliding buffer
       ↓
Mel-spectrogram (ONNX)
       ↓
Embedding model (ONNX)
       ↓
Wake word classifier (ONNX)
       ↓
Detection result (UI update via callback)
```

## Prerequisites

- Android Studio (latest stable version)
- Android device running Android 6.0 (API 23) or higher
- Internet access for downloading dependencies (e.g., ONNX Runtime Mobile)
- Place the following models in your `assets/` folder:
  - `melspectrogram.onnx`
  - `embedding_model.onnx`
  - `your_custom_wakeword_model.onnx` (e.g., `chaamiiya.onnx`)
  - `yamnet.tflite` and `yamnet_labels.txt` for sound classification

## Usage
1.    Clone this repository.
2.    Open the project in Android Studio.
3.    Place the required .onnx and .tflite models in the assets/ folder.
4.    Build and run the app on a real device or emulator (microphone access required).
5.    Speak the wake word and observe detection results in the UI.

## Notes
-    This project provides a fully on-device wake word detection pipeline using ONNX models — no server or internet required after setup.
-    The wake word model and embedding extractor are fully customizable; you can retrain models using Python and deploy them here.
