// Linear actuator extend / retract functions through an L298N (Motor A).
//
// Pin setup and drive logic are taken from actuator_button.
//
// Wiring: ENA -> D5 (PWM), IN1 -> D10, IN2 -> D8, shared GND with the L298N.
//
// There is no position sensor, so extend()/retract() run the motor for STROKE_MS
// (the time for a full stroke). Assumes the actuator starts fully retracted.

const int ENA = 5;   // PWM pin
const int IN1 = 10;
const int IN2 = 8;

const uint8_t SPEED = 255;                 // 0-255
const unsigned long STROKE_MS = 5000;      // time to travel full stroke; set to your actuator's
const float STROKE_M = 0.10;               // full stroke in meters = tube_prismatic_joint range (0..0.1)

enum Dir { STOPPED, UP, DOWN };

Dir currentDir = STOPPED;   // last direction commanded, used for position tracking
float posM = 0.0;           // estimated tube_prismatic_joint position (m); 0 = retracted

void drive(Dir d) {
  currentDir = d;
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

// Extend the actuator fully (blocks until done).
void extendActuator() {
  drive(UP);
  delay(STROKE_MS);
  drive(STOPPED);
  posM = STROKE_M;
}

// Retract the actuator fully (blocks until done).
void retractActuator() {
  drive(DOWN);
  delay(STROKE_MS);
  drive(STOPPED);
  posM = 0.0;
}

// "Excited" motion: switch direction every 2 seconds, up then down, for a few
// cycles (blocks until done). Each leg is shorter than a full stroke, so the
// actuator ends back where it started.
const unsigned long EXCITED_LEG_MS = 1500;
const int EXCITED_CYCLES = 3;

void excitedActuator() {
  for (int i = 0; i < EXCITED_CYCLES; i++) {
    drive(UP);
    delay(EXCITED_LEG_MS);
    drive(DOWN);
    delay(EXCITED_LEG_MS);
  }
  drive(STOPPED);
}

// ---- URDF animation playback: tube_prismatic_joint ----
// Keyframes from Urdf_Animation.effective-motion.json (timeMs, meters). Every
// tangent in that channel is flat (deltaValue 0), so segments are smoothstep curves.
struct Key { unsigned long t; float v; };
const Key TUBE_KEYS[] = {
  {14000, 0.05}, {17000, 0.00}, {18000, 0.05}, {21333, 0.05},
  {22667, 0.08}, {23667, 0.05}, {31000, 0.05}, {31667, 0.10},
  {32000, 0.02}, {32333, 0.10}, {33000, 0.00}, {33667, 0.10},
  {34333, 0.01}, {35000, 0.10}, {35667, 0.01}, {36333, 0.10},
  {37000, 0.05}
};
const int N_TUBE_KEYS = sizeof(TUBE_KEYS) / sizeof(TUBE_KEYS[0]);
const float DEADBAND_M = 0.002;   // stop within 2 mm of the target to avoid chatter

// Joint target (m) at animation time t (ms).
float tubeTargetAt(unsigned long t) {
  if (t <= TUBE_KEYS[0].t) return TUBE_KEYS[0].v;
  for (int i = 1; i < N_TUBE_KEYS; i++) {
    if (t <= TUBE_KEYS[i].t) {
      const Key &a = TUBE_KEYS[i - 1], &b = TUBE_KEYS[i];
      float u = (float)(t - a.t) / (float)(b.t - a.t);
      float s = u * u * (3.0 - 2.0 * u);
      return a.v + (b.v - a.v) * s;
    }
  }
  return TUBE_KEYS[N_TUBE_KEYS - 1].v;
}

// Advance the dead-reckoned position by dtMs at the current direction, then
// pick a direction toward targetM. Clamping to [0, STROKE_M] re-syncs the
// estimate whenever the actuator is pushed against an end stop.
void stepToward(float targetM, unsigned long dtMs) {
  if (currentDir == UP)   posM += STROKE_M * dtMs / STROKE_MS;
  if (currentDir == DOWN) posM -= STROKE_M * dtMs / STROKE_MS;
  posM = constrain(posM, 0.0, STROKE_M);

  float err = targetM - posM;
  Dir want = (err > DEADBAND_M) ? UP : (err < -DEADBAND_M) ? DOWN : STOPPED;
  // At an end stop with the target beyond it, stop instead of stalling the motor.
  if ((want == UP && posM >= STROKE_M) || (want == DOWN && posM <= 0.0)) want = STOPPED;
  if (want != currentDir) drive(want);
}

// Move to targetM and stop (blocks until there).
void moveToPosition(float targetM) {
  unsigned long last = millis();
  do {
    delay(5);
    unsigned long now = millis();
    stepToward(targetM, now - last);
    last = now;
  } while (currentDir != STOPPED);
}

// Play the tube_prismatic_joint animation (blocks until done, ~23 s).
// Assumes posM is accurate on entry (retracted after power-up).
void playTubeAnimation() {
  const unsigned long T0 = TUBE_KEYS[0].t;
  const unsigned long T_END = TUBE_KEYS[N_TUBE_KEYS - 1].t;

  moveToPosition(TUBE_KEYS[0].v);          // get to the animation's start pose

  unsigned long start = millis(), last = start;
  while (true) {
    unsigned long now = millis();
    unsigned long t = T0 + (now - start);
    if (t >= T_END) break;
    stepToward(tubeTargetAt(t), now - last);
    last = now;
    delay(5);
  }
  moveToPosition(TUBE_KEYS[N_TUBE_KEYS - 1].v);
  drive(STOPPED);
}

void setup() {
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  drive(STOPPED);
  Serial.begin(9600);
}

void loop() {
  // Serial commands from chatgpt_actuator.py: 'E' = extend, 'R' = retract, 'X' = excited.
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'E') {
      extendActuator();
      Serial.println("DONE extended");
    } else if (c == 'R') {
      retractActuator();
      Serial.println("DONE retracted");
    } else if (c == 'X') {
      excitedActuator();
      Serial.println("DONE excited");
    } else if (c == 'A') {
      playTubeAnimation();
      Serial.println("DONE animation");
    }
  }
}
