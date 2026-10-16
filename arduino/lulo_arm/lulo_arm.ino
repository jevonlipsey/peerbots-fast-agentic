// lulo_arm.ino -- USB serial controlled linear actuator for Lulo's body.
//
// the running python pipeline owns the usb port and sends newline-terminated
// ascii commands so lulo can physically react with his emotions. the onboard
// button stays wired up as a manual override, exactly like actuator_button.ino.
//
// commands (case-insensitive, one per line):
//   PING                 -> OK PONG
//   STATUS               -> STATUS {"pos":<0-100>,"dir":"UP|DOWN|STOPPED","gesture":"<name>","busy":0|1}
//   STOP                 -> stop immediately, cancel any running gesture
//   RESET                -> assume fully retracted (pos = 0) and stop
//   SPEED <0-255>        -> set pwm duty for subsequent moves
//   UP [ms] / DOWN [ms]  -> drive for ms, or continuously when ms is omitted
//   GESTURE <name>       -> play a canned gesture:
//                           stop rest extend retract nod bounce startle slump wave
//
// wiring: S1 button between D2 and GND (internal pull-up, pressed = LOW),
//         ENA -> D5 (PWM), IN1 -> D10, IN2 -> D8, shared GND with the L298N.
//
// there is no position sensor: position is integrated from drive time, assumes
// fully retracted at power-up, and re-syncs whenever an end is reached. the
// soft limits below stop the motor before it can drive past either end.

const int BTN = 2;
const int ENA = 5;   // PWM pin
const int IN1 = 10;
const int IN2 = 8;

uint8_t speed = 255;                       // 0-255

const unsigned long STROKE_MS   = 5000;    // time to travel full stroke; set to your actuator's
const unsigned long DEBOUNCE_MS = 30;
const unsigned long HOLD_MS     = 200;     // press longer than this counts as a hold (not a tap)
const unsigned long DOUBLE_MS   = 350;     // max gap between tap release and second press

enum Dir { STOPPED, UP, DOWN };

Dir dir = STOPPED;
unsigned long posMs = 0;                   // estimated extension, 0 = retracted .. STROKE_MS = extended
unsigned long lastUpdate = 0;

// timed move: drive until this millis() stamp, 0 = drive until STOP / limit
unsigned long moveEndMs = 0;

// debounce + manual button state
bool rawLast = HIGH;
bool pressed = false;
unsigned long lastRawChange = 0;
unsigned long pressStart = 0;
unsigned long lastTapRelease = 0;
bool lastWasTap = false;
bool reversed = false;
Dir holdDir = STOPPED;

// gesture player
struct Step { char d; unsigned int ms; unsigned int pause; };  // d: 'U','D','S'(hold)
struct Gesture { const char* name; const Step* steps; uint8_t count; };

const Step G_STOP[]    = { {'S', 0, 0} };
const Step G_REST[]    = { {'S', 0, 0} };
const Step G_EXTEND[]  = { {'U', STROKE_MS, 0} };
const Step G_RETRACT[] = { {'D', STROKE_MS, 0} };
const Step G_NOD[]     = { {'U', 180, 0}, {'D', 180, 0} };
const Step G_BOUNCE[]  = { {'U', 200, 60}, {'D', 200, 60}, {'U', 200, 60}, {'D', 200, 0} };
const Step G_STARTLE[] = { {'U', 450, 150}, {'D', 150, 0} };
const Step G_SLUMP[]   = { {'D', 700, 150}, {'D', 700, 0} };
const Step G_WAVE[]    = { {'U', 170, 40}, {'D', 170, 40}, {'U', 170, 40}, {'D', 170, 40}, {'U', 170, 40}, {'D', 170, 0} };

const Gesture GESTURES[] = {
  {"stop",    G_STOP,    1},
  {"rest",    G_REST,    1},
  {"extend",  G_EXTEND,  1},
  {"retract", G_RETRACT, 1},
  {"nod",     G_NOD,     2},
  {"bounce",  G_BOUNCE,  4},
  {"startle", G_STARTLE, 2},
  {"slump",   G_SLUMP,   2},
  {"wave",    G_WAVE,    6},
};
const uint8_t GESTURE_COUNT = sizeof(GESTURES) / sizeof(GESTURES[0]);

bool gActive = false;
const Step* gSteps = nullptr;
uint8_t gCount = 0;
uint8_t gIndex = 0;
uint8_t gPhase = 0;                        // 0 = moving, 1 = pausing
unsigned long gPhaseEnd = 0;
const char* gName = "stop";

char cmdBuf[48];
uint8_t cmdLen = 0;

void drive(Dir d) {
  // soft end stops: never drive further than the estimate allows
  if (d == UP && posMs >= STROKE_MS) d = STOPPED;
  if (d == DOWN && posMs == 0) d = STOPPED;

  dir = d;
  if (d == UP) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, speed);
  } else if (d == DOWN) {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    analogWrite(ENA, speed);
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

void cancelGesture() {
  gActive = false;
  gName = "stop";
}

void startStep() {
  const Step& s = gSteps[gIndex];
  Dir d = (s.d == 'U') ? UP : (s.d == 'D') ? DOWN : STOPPED;
  drive(d);
  gPhase = 0;
  gPhaseEnd = millis() + s.ms + 1;
}

void gestureStart(const Gesture& g) {
  gSteps = g.steps;
  gCount = g.count;
  gIndex = 0;
  gActive = true;
  gName = g.name;
  startStep();
}

void gestureUpdate() {
  if (!gActive) return;
  unsigned long now = millis();
  if (now < gPhaseEnd) return;

  const Step& s = gSteps[gIndex];
  if (gPhase == 0 && s.pause > 0) {
    drive(STOPPED);
    gPhase = 1;
    gPhaseEnd = now + s.pause;
    return;
  }
  gIndex++;
  if (gIndex >= gCount) {
    gActive = false;
    gName = "stop";
    drive(STOPPED);
    return;
  }
  startStep();
}

const Gesture* findGesture(const char* name) {
  for (uint8_t i = 0; i < GESTURE_COUNT; i++) {
    if (strcasecmp(GESTURES[i].name, name) == 0) return &GESTURES[i];
  }
  return nullptr;
}

void printStatus() {
  int pct = (int)((posMs * 100UL) / STROKE_MS);
  const char* d = (dir == UP) ? "UP" : (dir == DOWN) ? "DOWN" : "STOPPED";
  Serial.print(F("STATUS {\"pos\":"));
  Serial.print(pct);
  Serial.print(F(",\"dir\":\""));
  Serial.print(d);
  Serial.print(F("\",\"gesture\":\""));
  Serial.print(gName);
  Serial.print(F("\",\"busy\":"));
  Serial.print((dir != STOPPED || gActive) ? 1 : 0);
  Serial.println(F("}"));
}

void handleCommand(char* line) {
  // split into verb + optional argument
  char* sp = strchr(line, ' ');
  char* arg = nullptr;
  if (sp) {
    *sp = '\0';
    arg = sp + 1;
    while (*arg == ' ') arg++;
  }

  if (strcasecmp(line, "PING") == 0) {
    Serial.println(F("OK PONG"));
  } else if (strcasecmp(line, "STATUS") == 0) {
    printStatus();
  } else if (strcasecmp(line, "STOP") == 0) {
    cancelGesture();
    moveEndMs = 0;
    drive(STOPPED);
    Serial.println(F("OK STOP"));
  } else if (strcasecmp(line, "RESET") == 0) {
    cancelGesture();
    moveEndMs = 0;
    drive(STOPPED);
    posMs = 0;
    Serial.println(F("OK RESET"));
  } else if (strcasecmp(line, "SPEED") == 0) {
    if (arg) {
      long v = atol(arg);
      speed = (uint8_t)constrain(v, 0, 255);
      Serial.println(F("OK SPEED"));
    } else {
      Serial.println(F("ERR SPEED needs 0-255"));
    }
  } else if (strcasecmp(line, "UP") == 0 || strcasecmp(line, "DOWN") == 0) {
    Dir d = (strcasecmp(line, "UP") == 0) ? UP : DOWN;
    cancelGesture();
    unsigned long ms = arg ? strtoul(arg, nullptr, 10) : 0;
    drive(d);
    moveEndMs = ms ? (millis() + ms) : 0;
    Serial.println(F("OK MOVE"));
  } else if (strcasecmp(line, "GESTURE") == 0) {
    const Gesture* g = arg ? findGesture(arg) : nullptr;
    if (g) {
      moveEndMs = 0;
      gestureStart(*g);
      Serial.print(F("OK GESTURE "));
      Serial.println(g->name);
    } else {
      Serial.println(F("ERR unknown gesture"));
    }
  } else if (line[0] != '\0') {
    Serial.println(F("ERR unknown command"));
  }
}

void pollSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      cmdBuf[cmdLen] = '\0';
      handleCommand(cmdBuf);
      cmdLen = 0;
    } else if (c != '\r' && cmdLen < sizeof(cmdBuf) - 1) {
      cmdBuf[cmdLen++] = c;
    }
  }
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

void pollButton() {
  bool raw = digitalRead(BTN);
  if (raw != rawLast && millis() - lastRawChange > DEBOUNCE_MS) {
    lastRawChange = millis();
    rawLast = raw;
    pressed = (raw == LOW);
    if (pressed) {
      // manual press takes over from any running gesture
      cancelGesture();
      moveEndMs = 0;
      onPress();
    } else {
      onRelease();
    }
  }

  if (pressed && dir == STOPPED && (reversed || millis() - pressStart >= HOLD_MS)) {
    drive(holdDir);
  }
}

void setup() {
  pinMode(BTN, INPUT_PULLUP);
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  drive(STOPPED);
  lastUpdate = millis();

  Serial.begin(115200);
  // uno/nano reset when the port opens; give the host a moment, then say hello
  delay(300);
  Serial.println(F("OK LULO_ARM READY"));
}

void loop() {
  updatePosition();
  gestureUpdate();

  // stop a timed move once its window elapses
  if (dir != STOPPED && moveEndMs != 0 && millis() >= moveEndMs) {
    moveEndMs = 0;
    drive(STOPPED);
  }

  pollButton();
  pollSerial();
}
