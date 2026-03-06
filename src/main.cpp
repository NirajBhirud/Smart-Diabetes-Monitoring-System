#include "esp_system.h"
#include <Wire.h>
#include "MAX30100_PulseOximeter.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_LIS3DH.h>
#include <Adafruit_Sensor.h>
#include <WiFi.h>
#include <HTTPClient.h>

// ─── WiFi / Server ────────────────────────────────────────────────────────────
const char* WIFI_SSID = "motoedge";
const char* WIFI_PASS = "PVMahajan";
#define FLASK_URL "http://10.149.123.7:5000/iot/predict"

// ─── Hardware ─────────────────────────────────────────────────────────────────
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET   -1

// ─── Timing & Thresholds ──────────────────────────────────────────────────────
#define REPORTING_PERIOD_MS  1000   // OLED/serial refresh
#define STEP_THRESHOLD       12.0f  // m/s² magnitude threshold
#define STEP_DEBOUNCE        300    // ms between valid steps
#define FINGER_TIMEOUT       3000   // ms before "no finger" declared
#define STABLE_READS_NEEDED  5      // consecutive valid reads before accepting HR/SpO2
#define FLASK_SEND_INTERVAL  5000   // send to Flask every 5 s (was 1 s → caused instant reset)

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
PulseOximeter pox;
Adafruit_LIS3DH lis = Adafruit_LIS3DH();

// ─── State ────────────────────────────────────────────────────────────────────
uint32_t lastReport     = 0;
uint32_t lastBeatTime   = 0;
uint32_t lastStepTime   = 0;
uint32_t lastFlaskSend  = 0;
unsigned long measurementStart = 0;
bool measurementActive = false;


float stableHR = 0;
float stableSpO2 = 0;
float heartRate  = 0;
float spo2       = 0;
float glucose    = 0;
int   steps      = 0;
int   stableCount = 0;        // FIX #3 – stability counter
bool  wasFingerValid = false;
bool  dataSent = false;       // FIX #1 – guard against re-sending / premature reset



// ─── Helpers ──────────────────────────────────────────────────────────────────

void connectWiFi()
{
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to WiFi");
  uint8_t attempts = 0;
  const uint8_t maxAttempts = 40;
 while (WiFi.status() != WL_CONNECTED && attempts < maxAttempts) // FIX #2 – timeout, don't hang forever
  {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED)
  {
    Serial.println("\nWiFi Connected: " + WiFi.localIP().toString());
  }
  else
  {
    Serial.println("\nWiFi FAILED – continuing offline");
  }
}

// FIX #5 – glucose formula guard (spo2 must be physiologically valid)
float calcGlucose(float hr, float sp)
{
  if (hr < 40 || hr > 220 || sp < 85.0f || sp > 100.0f) return 0;
  return 16714.61f + 0.47f * hr - 351.045f * sp + 1.85f * (sp * sp);
}

void sendToFlask()
{
  if (WiFi.status() != WL_CONNECTED) return;
  if (heartRate < 50 || heartRate > 120) return;
  if (spo2 < 90 || spo2 > 100) return;
  if (glucose <= 0) return;  // FIX #6 – never send garbage data

  HTTPClient http;
  http.begin(FLASK_URL);
  http.setTimeout(5000);
  http.addHeader("Content-Type", "application/json");

  String payload = "{";
  payload += "\"Glucose\":"  + String(glucose,   1) + ",";
  payload += "\"HeartRate\":" + String(heartRate, 1) + ",";
 int activity;

if (steps < 10) activity = 0;
else if (steps < 40) activity = 1;
else activity = 2;

payload += "\"Activity\":" + String(activity) + ",";
  payload += "\"SpO2\":"      + String(spo2,      1);
  payload += "}";

  Serial.println("→ Sending: " + payload);
  int code = http.POST(payload);
  Serial.println("← HTTP " + String(code));
  http.end();


  // FIX #1 – REMOVED esp_restart() from here.
  // Restarting immediately after one send broke all subsequent readings.
  // If a hard reset is truly needed, do it only on explicit user action or after
  // a long idle period (see loop below).
}

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup()
{
  Serial.begin(115200);
  Wire.begin(21, 22);

  connectWiFi();  // FIX #7 – moved WiFi before hardware init so IP is ready for debug prints

  // OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C))
  {
    Serial.println("OLED NOT FOUND");
    while (1);
  }
  display.setTextColor(WHITE);

  // MAX30100
  if (!pox.begin())
  {
    Serial.println("MAX30100 NOT FOUND");
    while (1);
  }
  pox.setIRLedCurrent(MAX30100_LED_CURR_24MA);

  // LIS3DH
  if (!lis.begin(0x18))
  {
    if (!lis.begin(0x19))
    {
      Serial.println("LIS3DH NOT FOUND");
      while (1);
    }
  }
  lis.setRange(LIS3DH_RANGE_4_G);

  Serial.println("System Ready");
}

// ─── Loop ─────────────────────────────────────────────────────────────────────
void loop()
{
  pox.update();   // always run

  if (millis() < 5000) return;

  

  // ── Step Detection ──────────────────────────────────────────────────────────
  sensors_event_t event;
  lis.getEvent(&event);

  float magnitude = sqrt(
    event.acceleration.x * event.acceleration.x +
    event.acceleration.y * event.acceleration.y +
    event.acceleration.z * event.acceleration.z
  );

  if (magnitude > STEP_THRESHOLD && (millis() - lastStepTime) > STEP_DEBOUNCE)
  {
    steps++;
    lastStepTime = millis();
  }

  // ── Periodic Reporting ──────────────────────────────────────────────────────
  if (millis() - lastReport > REPORTING_PERIOD_MS)
  {
    float rawHR   = pox.getHeartRate();
    float rawSpO2 = pox.getSpO2();

    // A reading is "live" only if HR and SpO2 are in valid range
    bool currentlyValid = (rawHR > 45 && rawHR < 120 && rawSpO2 > 85);

    if (currentlyValid)
    {
      lastBeatTime = millis();                                  // refresh the heartbeat timestamp
      stableCount  = min(stableCount + 1, STABLE_READS_NEEDED);
    }
    else
    {
      stableCount = 0;   // immediately collapse stability on ANY invalid reading
    }

    // fingerValid requires BOTH: sensor currently good AND not timed out
    bool timedOut    = (millis() - lastBeatTime > FINGER_TIMEOUT);
    bool fingerValid = currentlyValid && !timedOut && (stableCount >= STABLE_READS_NEEDED);

    // ── Finger Removed → Reset Sensor (one-time transition) ──────────────
    if (!fingerValid && wasFingerValid)
    {
      Serial.println("Finger removed → Resetting MAX30100");
      pox.begin();
      pox.setIRLedCurrent(MAX30100_LED_CURR_24MA);
      stableCount = 0;
      dataSent    = false;
    }
    wasFingerValid = fingerValid;

    // ── Always update values based on current finger state ────────────────
    if (fingerValid)
{
    heartRate = rawHR;
    spo2      = rawSpO2;
    glucose   = calcGlucose(heartRate, spo2);

    if (!measurementActive)
    {
    measurementStart = millis();
    measurementActive = true;
  }
}
    else
    {
      // Finger not present → force all health values to 0 every cycle
      heartRate = 0;
      spo2      = 0;
      glucose   = 0;
    }

    // ── Serial ──────────────────────────────────────────────────────────────
    Serial.printf("HR: %.1f | SpO2: %.1f | Glucose: %.1f | Steps: %d\n",
                  heartRate, spo2, glucose, steps);

    // ── OLED ────────────────────────────────────────────────────────────────
    display.clearDisplay();
    display.setTextSize(1);

    display.setCursor(0, 0);
    display.print("HR: ");
    display.println(String(heartRate, 1));      // directly shows 0.0 when finger removed

    display.setCursor(0, 15);
    display.print("SpO2: ");
    display.println(String(spo2, 1));           // directly shows 0.0 when finger removed

    display.setCursor(0, 30);
    display.print("Glucose: ");
    display.print(String(glucose, 1));          // directly shows 0.0 when finger removed
    display.print(" mg/dL");

    display.setCursor(0, 45);
    display.print("Steps: ");
    display.println(steps);

    if (!fingerValid)
    {
      display.setCursor(0, 57);
      display.setTextSize(1);
      display.println("Place Finger...");
    }

    display.display();
    lastReport = millis();

    // ── Flask Send ──────────────────────────────────────────────────────────
    // FIX #1 + FIX #6 – only send when stable, valid, not already sent, and interval elapsed
    // ── Flask Send ─────────────────────────────────────────
if (fingerValid && (millis() - lastFlaskSend) > FLASK_SEND_INTERVAL)
{
  sendToFlask();
  lastFlaskSend = millis();
}

// Reset values after 5 seconds of measurement
if (measurementActive && millis() - measurementStart > 5000)
{
  Serial.println("Measurement complete → Resetting values");

  heartRate = 0;
  spo2 = 0;
  glucose = 0;

  stableCount = 0;
  measurementActive = false;

  pox.begin();
  pox.setIRLedCurrent(MAX30100_LED_CURR_24MA);
}
  }
}