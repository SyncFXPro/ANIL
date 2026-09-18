# ANIL

Evaluating Identity Reconstruction Through Pitch-Shift Prediction and Inversion

This repo starts with vector similarity only. Audio files and CNNs come later.

## Why vectors first

A sound clip can be treated as a vector. One second of audio at 44.1 kHz has shape `(44100,)`. You cannot plot that many dimensions, but cosine similarity still works:

```
cos(theta) = (F1 · F2) / (||F1|| ||F2||)
```

- Close to `1`: the vectors point in almost the same direction.
- Around `0`: the directions are very different.
- Near `-1`: the vectors point opposite ways.

This project shows that math on two tiny 3-D examples, then plots those three dimensions.

Example vectors:

```
F1 = [1.0, 2.0, 3.0]
F2 = [1.2, 1.8, 2.9]
```

## Project layout

```
ANIL/
    main.py
    requirements.txt
    README.md
```

## Setup

```
python -m pip install -r requirements.txt
```

## Run

```
python main.py
```

You should see the two vectors printed, their cosine similarity, and a 3-D quiver plot.
