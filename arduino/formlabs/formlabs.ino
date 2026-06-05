#include "ArduinoBLE.h"
#include "Arduino_BMI270_BMM150.h"

#define SVC "fa9b1d2c-3e4f-4a5b-9c6d-7e8f9a0b1c2d"
#define CHR "fa9b1d2c-3e4f-4a5b-9c6d-7e8f9a0b1c2e"
#define LEFT  0xf2bec62266b67b61ULL
#define RIGHT 0x2a1159d4adc51cdbULL

BLEService svc(SVC);
BLECharacteristic chr(CHR, BLERead | BLENotify, 40);

void setup() {
  IMU.begin();

  pinMode(A3, INPUT_PULLUP);
  pinMode(A5, INPUT_PULLUP);
  pinMode(D4, INPUT_PULLUP);
  pinMode(D6, INPUT_PULLUP);

  uint64_t uid = ((uint64_t)NRF_FICR->DEVICEID[1] << 32) | NRF_FICR->DEVICEID[0];
  const char* name = uid == LEFT ? "left" : uid == RIGHT ? "right" : "unknown";

  BLE.begin();
  BLE.setLocalName(name);
  BLE.setDeviceName(name);
  BLE.setAdvertisedService(svc);
  svc.addCharacteristic(chr);
  BLE.addService(svc);
  BLE.advertise();
}

void loop() {
  BLE.poll();

  static float ax = 0, ay = 0, az = 0;
  static float gx = 0, gy = 0, gz = 0;
  static float mx = 0, my = 0, mz = 0;

  if (IMU.accelerationAvailable()) IMU.readAcceleration(ax, ay, az);
  if (IMU.gyroscopeAvailable()) IMU.readGyroscope(gx, gy, gz);
  if (IMU.magneticFieldAvailable()) IMU.readMagneticField(mx, my, mz);

  struct __attribute__((packed)) {
    float gx, gy, gz;
    float ax, ay, az;
    float mx, my, mz;
    bool ba, bb, bc, bd;
  } packet = {
    gx, gy, gz,
    ax, ay, az,
    mx, my, mz,
    digitalRead(A3), digitalRead(A5),
    digitalRead(D4), digitalRead(D6),
  };

  chr.writeValue((uint8_t*)&packet, sizeof(packet));
}
