# AI basics — a running glossary for this project

> Plain-language notes so the whole pipeline makes sense. Grows as we go.

## The one big idea
A **model** isn't hand-written rules. We show it thousands of examples (Nepali text + a human
saying it) and it **adjusts itself** until it can do the same on new text. That self-adjustment =
**training**. The model is a big grid of numbers (**weights**) that start random and become meaningful.

Three ingredients, always:
- **Data** — Nepali audio + matching text (the textbook)
- **Model** — a neural network, here **VITS** (the student's brain)
- **Training** — running data through it on the GPU (studying)

## The TTS pipeline (text → sound)
1. **Frontend** — clean the text + convert to **phonemes** (pronunciation units). espeak-ng does this. *Where Nepali accuracy is won/lost.*
2. **Acoustic model** — trained neural net turns phonemes into a **spectrogram** (a picture of sound).
3. **Vocoder** — turns that picture into real audio waves.
VITS does steps 2+3 in one network.

## Training loop
show sentence → model guesses audio → compare to real recording (**loss** = how wrong) →
nudge weights to reduce loss → repeat hundreds of thousands of times. Static → human over time.

## Vocabulary
| Term | Plain meaning |
|---|---|
| GPU | Chip with thousands of tiny calculators working at once; makes AI math fast. Ours: RTX 5050. |
| CUDA | The language PyTorch uses to drive an NVIDIA GPU. "cu128" = version for Blackwell chips. |
| PyTorch | The engine (Python library) that builds models + runs training on the GPU. |
| venv | Sealed box for this project's Python tools. |
| WSL | Real Linux inside Windows; where AI tools run best. |
| weights | The numbers inside the model that get adjusted during training. |
| loss | A score of how wrong the model's guess is; training drives it down. |
| phoneme | A unit of pronunciation (a sound), e.g. /n/, /eː/. |
| spectrogram | A picture of sound (frequency over time) the model predicts. |
| vocoder | Converts a spectrogram into actual audio. |
| epoch / step | One pass of training data / one single weight update. |
| checkpoint | A saved snapshot of the model partway through training. |
| inference | Using a trained model to generate new output (the opposite of training). |
| VITS | The specific TTS model architecture we use (fast, good, supports multi-speaker). |
| Piper | The toolkit that trains VITS and exports it to run offline. |
| MMS | Meta's multilingual speech models; `mms-tts-npi` is their Nepali voice (our baseline). |
| warm-start | Begin training from someone else's finished model instead of random — much faster. |
