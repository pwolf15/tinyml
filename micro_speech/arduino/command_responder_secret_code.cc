/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include "command_responder.h"

#include "Arduino.h"

// Toggles the built-in LED every inference, and lights a colored LED depending
// on which word was detected.
void RespondToCommand(tflite::ErrorReporter* error_reporter,
                      int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
  static bool is_initialized = false;
  static char last_three[3] = { '0', '0', '0' };
  static char code[3]       = { 'y', 'y', 'y' };

  if (!is_initialized) {
    pinMode(LED_BUILTIN, OUTPUT);
    // Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    // Ensure the LED is off by default.
    // Note: The RGB LEDs on the Arduino Nano 33 BLE
    // Sense are on when the pin is LOW, off when HIGH.
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
    is_initialized = true;
  }
  static int32_t last_command_time = 0;
  static int count = 0;
  static int certainty = 220;

  if (is_new_command) {
    TF_LITE_REPORT_ERROR(error_reporter, "Heard %s (%d) @%dms", found_command,
                         score, current_time);

    // get last three recognized characters
    last_three[0] = last_three[1];
    last_three[1] = last_three[2];
    last_three[2] = found_command[0];

    // If we hear a command, light up the appropriate LED
    if (found_command[0] == 'y') {
      last_command_time = current_time;
      digitalWrite(LEDB, LOW);  // Green for yes
    }

    if (found_command[0] == 'n') {
      last_command_time = current_time;
      digitalWrite(LEDR, LOW);  // Red for no
    }

    if (found_command[0] == 'u') {
      last_command_time = current_time;
      // no light
    }

    if (last_three[0] == code[0] && last_three[1] == code[1] && last_three[2] == code[2]) {
      digitalWrite(LEDG, LOW);  // Blue for unknown
    }
  }

  // If last_command_time is non-zero but was >3 seconds ago, zero it
  // and switch off the LED.
  if (last_command_time != 0) {
    if (last_command_time < (current_time - 3000)) {
      last_command_time = 0;
      digitalWrite(LED_BUILTIN, LOW);
      digitalWrite(LEDR, HIGH);
      digitalWrite(LEDG, HIGH);
      digitalWrite(LEDB, HIGH);
    }
    // If it is non-zero but <3 seconds ago, do nothing.
    return;
  }

  // Otherwise, toggle the LED every time an inference is performed.
  ++count;
  if (count & 1) {
    digitalWrite(LED_BUILTIN, HIGH);
  } else {
    digitalWrite(LED_BUILTIN, LOW);
  }
}

#endif  // ARDUINO_EXCLUDE_CODE