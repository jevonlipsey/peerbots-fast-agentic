// One-button linear actuator control through an L298N (Motor A).
//
//   Hold              -> actuator moves. Default direction is UP (extend), unless it is
//                        fully extended, in which case the default is DOWN (retract).
//   Double click+hold -> tap, then press again quickly and hold: moves the OPPOSITE of the
//                        default direction for that hold (even if only partly extended).
//
// Wiring: S1 button between D2 and GND (internal pull-up, pressed = LOW),
//         ENA -> D5 (PWM), IN1 -> D10, IN2 -> D8, shared GND with the L298N.
//
// There is no position sensor, so position is estimated from how long the actuator has
// been driven. It assumes the actuator is fully retracted at power-up/reset, and the
// estimate re-syncs whenever it reaches either end.

const int BTN = 2;
const int ENA = 5;   // PWM pin
const int IN1 = 10;
const int IN2 = 8;                                                                                                                                      

const uint8_t SPEED = 255;                 // 0-255

const unsigned long STROKE_MS   = 5000;    // time to travel full stroke; set to your actuator's
const unsigned long DEBOUNCE_MS = 30;
const unsigned long HOLD_MS     = 200;     // press longer than this counts as a hold (not a tap)
const unsigned long DOUBLE_MS   = 350;     // max gap between tap release and second press

enum Dir { STOPPED, UP, DOWN };

Dir dir = STOPPED;
unsigned long posMs = 0;                   // estimated extension, 0 = retracted .. STROKE_MS = extended
unsigned long lastUpdate = 0;

bool rawLast = HIGH;                       // debounce state
bool pressed = false;
unsigned long lastRawChange = 0;

unsigned long pressStart = 0;              // millis() when the current press began
unsigned long lastTapRelease = 0;
bool lastWasTap = false;
bool reversed = false;                     // this press is the second click of a double click
Dir holdDir = STOPPED;                     // direction chosen for the current hold

void drive(Dir d) {
  dir = d;
  if (d == UP) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, SPEED);
  } else if (d == DOWN) {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    analogWrite(ENA, SPEED);
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, 0);
  }
}

// Integrate estimated position while the motor runs.
void updatePosition() {
  unsigned long now = millis();
  unsigned long dt = now - lastUpdate;
  lastUpdate = now;
  if (dir == UP) {
    posMs = min(STROKE_MS, posMs + dt);
  } else if (dir == DOWN) {
    posMs = (dt > posMs) ? 0 : posMs - dt;
  }
}

Dir defaultDirection() {
  return (posMs >= STROKE_MS) ? DOWN : UP;
}

void onPress() {
  unsigned long now = millis();
  pressStart = now;
  reversed = lastWasTap && (now - lastTapRelease <= DOUBLE_MS);
  Dir def = defaultDirection();
  holdDir = reversed ? (def == UP ? DOWN : UP) : def;
}

void onRelease() {
  unsigned long now = millis();
  lastWasTap = !reversed && (now - pressStart < HOLD_MS);
  lastTapRelease = now;
  reversed = false;
  drive(STOPPED);
}

void setup() {
  pinMode(BTN, INPUT_PULLUP);
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  drive(STOPPED);
  lastUpdate = millis();
}

void loop() {
  updatePosition();

  // Debounced button edges.
  bool raw = digitalRead(BTN);
  if (raw != rawLast && millis() - lastRawChange > DEBOUNCE_MS) {
    lastRawChange = millis();
    rawLast = raw;
    pressed = (raw == LOW);
    if (pressed) onPress(); else onRelease();
  }

  // While held: second click of a double click moves immediately, a normal press
  // waits HOLD_MS so a quick tap doesn't jerk the actuator.
  if (pressed && dir == STOPPED && (reversed || millis() - pressStart >= HOLD_MS)) {
    drive(holdDir);
  }
}
