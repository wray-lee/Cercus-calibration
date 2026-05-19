/*
 -------------------------------------------------------------------------------------
  BioMoR — Dual Optical Sensor Calibration Firmware (Arduino Mega 2560)

  ML Calibration Dedicated Build
  ===============================
  Stripped-down firmware for high-fidelity 4-DOF optical flow data acquisition.
  Streams raw dual-axis displacement from two ADNS2083 sensors at 200 Hz,
  producing a 5-column feature matrix for downstream ML error calibration
  against camera vision ground truth.

  Output Protocol (per epoch, 5 ms window):
    t_ard, x_dx, x_dy, y_dx, y_dy\n

  Constraints
  -----------
  - ZERO delay() calls — hard real-time, non-blocking throughout.
  - 200 Hz (5 ms epoch) deterministic sampling via millis() guard.
  - All four optical axes accumulated per epoch — no dimension discarded.
  - 115200 baud, pure data stream — no command parsing, no stimulus outputs.

  Hardware Pin Map (Sensor Only)
  ------------------------------
    Pin  2  = SCLK Sensor X
    Pin  3  = SDIO Sensor X
    Pin 31  = SCLK Sensor Y
    Pin 33  = SDIO Sensor Y
 -------------------------------------------------------------------------------------
*/
#include <Arduino.h>
#include "ADNS2083.h"

// ==========================================================================
// SPI — Optical Sensors
// ==========================================================================
#define SCLK_X 2
#define SDIO_X 3
#define SCLK_Y 31
#define SDIO_Y 33

ADNS2083 OpticalX = ADNS2083(SCLK_X, SDIO_X);
ADNS2083 OpticalY = ADNS2083(SCLK_Y, SDIO_Y);

// ==========================================================================
// 200 Hz Deterministic Streaming State
// ==========================================================================
unsigned long microsPre = 0;
const unsigned long MICROS_FRM = 5000; // 200 Hz = 5000 us epoch

// ==========================================================================
// setup()
// ==========================================================================
void setup()
{
    Serial.begin(115200);
    OpticalX.begin();
    OpticalY.begin();

    microsPre = micros();
}

// ==========================================================================
// loop()
// ==========================================================================
void loop()
{
    unsigned long microsNow = micros();

    // strictly triggered by 5000us
    if (microsNow - microsPre >= MICROS_FRM)
    {
        microsPre += MICROS_FRM;

        long x_dx = 0, x_dy = 0, y_dx = 0, y_dy = 0;

        if (OpticalX.motion())
        {
            x_dx = OpticalX.dx();
            x_dy = OpticalX.dy();
        }
        if (OpticalY.motion())
        {
            y_dx = OpticalY.dx();
            y_dy = OpticalY.dy();
        }

        Serial.print(millis());
        Serial.print(',');
        Serial.print(x_dx);
        Serial.print(',');
        Serial.print(x_dy);
        Serial.print(',');
        Serial.print(y_dx);
        Serial.print(',');
        Serial.println(y_dy);
    }
}