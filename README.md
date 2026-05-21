# Cercus-Calibration

![C++](https://img.shields.io/badge/C%2B%2B-00599C?logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![PlatformIO](https://img.shields.io/badge/PlatformIO-F5822A?logo=platformio&logoColor=white)

Dual optical sensor calibration firmware and ML toolchain under the BioMoR architecture. Acquires high-frequency 4-DOF optical flow data from two ADNS2083 sensors and trains an autoencoder to reduce it to a calibrated 3-DOF motion matrix.

Ground truth calibration is derived entirely from camera-based visual processing. No treadmill sensors are involved.

## Architecture

```
+---------------------+         USB Serial (115200)         +---------------------------+
|  Arduino Mega 2560  | --------------------------------> |   Python GUI (Tkinter)    |
|                     |   t,x_dx,x_dy,y_dx,y_dy @200Hz   |                           |
|  ADNS2083 (X)  SPI  |                                   |   SerialReader daemon     |
|  ADNS2083 (Y)  SPI  |                                   |         |                 |
+---------------------+                                   |   collect_YYYYMMDD.csv    |
                                                           |         |                 |
                                                           |   Calibrator (PyTorch)    |
                                                           |   4-DOF -> 3-DOF AE      |
                                                           |         |                 |
                                                           |   calibration_cfg.json    |
                                                           +---------------------------+
```

## Hardware Setup

**MCU:** Arduino Mega 2560

**Pin Map (bit-banged SPI):**

| Signal | Pin | Sensor |
|--------|-----|--------|
| SCLK_X | 2   | Sensor X |
| SDIO_X | 3   | Sensor X |
| SCLK_Y | 31  | Sensor Y |
| SDIO_Y | 33  | Sensor Y |

**Real-time constraints:**
- 200 Hz (5 ms) hard sampling interval enforced by `micros()` accumulator.
- Zero `delay()` calls in the main loop. Fully non-blocking.
- Firmware streams a pure data pipe -- no inbound command parsing.

**Serial output format (115200 baud):**
```
millis,x_dx,x_dy,y_dx,y_dy
```

Build and upload with PlatformIO:

```bash
cd Hardware
pio run --target upload
```

## Software Installation

Python 3.9+ required. Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:
- `customtkinter>=5.2.0` -- GUI framework
- `pyserial>=3.5` -- serial communication
- `torch>=2.0.0` -- autoencoder training

## Usage Pipeline

```bash
cd ML
python main.py
```

The GUI drives a four-state machine: **IDLE → COLLECTING → TRAINING → DONE**.

1. **Connect** -- Select COM port and baud rate (115200), click Connect. Status dot turns green when linked.
2. **Collect** -- Click Start Collection. Sensor readings stream live. CSV file `collect_YYYYMMDD_HHMMSS.csv` is written incrementally. Click Stop when sufficient data is gathered (minimum 10 samples).
3. **Train** -- Click Start Training. The autoencoder runs for up to 10,000 epochs with early stopping. Loss and epoch progress are logged in real time.
4. **Export** -- On completion, a 3x3 calibration matrix is written to `calibration_cfg.json`.

## Algorithm Notes

The autoencoder maps raw 4-DOF sensor readings `[x_dx, x_dy, y_dx, y_dy]` to a 3-DOF latent space `[X, Y, Z]` representing calibrated physical motion.

**Encoder** (4 → 3): `z = W_enc @ s`, where `W_enc` is a 3x4 weight matrix initialized with physical priors that assign each dominant sensor axis to one latent dimension.

**Decoder** (3 → 4): `s' = W_dec @ z`, reconstructing the original 4-DOF signal for validation.

**Loss function:**

```
L = MSE_recon / var(S) + lambda * ||W_enc * mask||_2^2
```

- **Scale-invariant reconstruction**: normalized MSE decouples calibration accuracy from signal amplitude.
- **L2 ridge penalty**: applied only to off-diagonal encoder weights via a binary mask, suppressing cross-axis crosstalk while leaving principal axis weights unconstrained.
- **Anchoring**: diagonal encoder weights are hard-locked after each optimizer step to prevent scale drift.

**Data cleaning**: MAD-based (median absolute deviation) adaptive thresholding filters out silent frames and outlier spikes before training.

**Output**: The trained encoder matrix is column-reindexed to produce a 3x3 production calibration matrix exported to `calibration_cfg.json`.
