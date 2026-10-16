# ASL Sign Language Translator

A real-time American Sign Language (ASL) translator that uses your webcam. It recognizes signs with a hand-landmark LSTM model and turns the resulting ASL gloss into an English sentence.

This project was completed through the **Michigan Data Science Team (MDST)** in Winter 2026. It is based on the original MDST repository:
[MichiganDataScienceTeam/W26-Sign-Language-Translator](https://github.com/MichiganDataScienceTeam/W26-Sign-Language-Translator)

## What it does

1. **Hand tracking:** MediaPipe detects up to two hands in each webcam frame and returns 21 landmarks per hand.
2. **Feature extraction:** Landmarks are normalized relative to the wrist and hand size, giving 84 features per frame (2 hands x 21 points x 2 coordinates). This makes the model less sensitive to where you sit or how large your hands appear.
3. **Sign classification:** A PyTorch LSTM (4 layers, hidden size 256) reads a rolling window of frames and predicts one of 20 ASL words.
4. **Sentence building:** Accepted signs are collected into an ASL gloss sequence.
5. **Translation:** After a short pause, the gloss is sent to an LLM that rewrites it as a natural English sentence.

The interface is a Streamlit app showing the live video with a hand skeleton overlay, the top-3 predictions with confidence scores, the sign history, and the English translation.

## Supported signs

Trained on 20 glosses from the [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) dataset:

BASKETBALL, BEE, BELT, BITE, BREAKFAST, CHRISTMAS, DEAF, DEMAND, DOG, EAT, FINE, FOREIGNER, HOSPITAL, MECHANIC, MOVIE, PARTY, PATIENT, ROCKINGCHAIR, SHAVE, WHATFOR

## Model

| Component | Details |
|---|---|
| Input | Sequence of frames, 84 features each |
| Recurrent layers | LSTM, 4 layers, hidden size 256, dropout 0.5 |
| Output head | Last timestep, then Linear 256 to 128, then Linear 128 to 20 |
| Loss | Cross-entropy with inverse-frequency class weights |
| Optimizer | AdamW (lr 1e-4, weight decay 1e-4), 50 epochs |

In the app, predictions use a 45-frame rolling window, require at least 10 frames with hands visible, and are accepted at 65% confidence or higher.

## Project structure

```
sign_language_translator/
  stlit_new.py                 # Streamlit app (final product)
  model_def.py                 # Video_LSTM_morelayers model class
  asl_citizen_processor.py     # MediaPipe landmark extraction and normalization
  train_asl_citizen.py         # Training script
  saved_models/                # Trained weights (.pth)
  asl_citizen_processed/       # Label map and processed config
```

## Limitations

- Only 20 words are supported.
- The model was trained on full single-sign clips but runs on a live rolling window, so accuracy depends on lighting, framing, and clear pauses between signs.

## Acknowledgments

Built with the Michigan Data Science Team. Original repository: [MichiganDataScienceTeam/W26-Sign-Language-Translator](https://github.com/MichiganDataScienceTeam/W26-Sign-Language-Translator). Uses the ASL Citizen dataset, MediaPipe, and PyTorch.
