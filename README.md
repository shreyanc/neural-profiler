# Neural Profiler

**Neural approach to profiling audio hardware**

---

### Overview

**Neural Profiler** explores novel machine learning architectures for characterizing and modeling audio hardware behavior — with a focus on **compression, coloration, and dynamic response**. Instead of traditional DSP-based profiling, this project leverages deep sequence models to learn differentiable mappings between clean and processed audio.

### Goals

* Develop data-driven models for **hardware compressor profiling**
* Investigate **previously unexplored neural architectures** for audio response modeling
* Evaluate **xLSTM** as a foundational model for capturing long-term temporal dependencies in dynamic audio systems

### Roadmap

* [ ] Data acquisition and preprocessing pipeline
* [ ] Baseline compressor response dataset
* [ ] xLSTM training for hardware-to-software response mapping
* [ ] Comparative evaluation vs. existing DSP emulation methods

### Tech Stack

* **Python**, **PyTorch**, **xLSTM**, **Librosa**, **NumPy**

---

> Early-stage research project — experimental and evolving.
