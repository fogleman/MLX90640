#include <Wire.h>

#define MLX90640_ADDR 0x33

#define CMD_READ 0
#define CMD_WRITE 1
#define CMD_READ_MANY 2
#define CMD_READ_CHESS_PAGE 3
#define CMD_READ_INTERLEAVED_PAGE 4

void read(uint16_t addr) {
    Wire.beginTransmission(MLX90640_ADDR);
    Wire.write(addr >> 8);
    Wire.write(addr & 0xff);
    Wire.endTransmission(false);
    Wire.requestFrom(MLX90640_ADDR, 2);
    Serial.write(Wire.read());
    Serial.write(Wire.read());
}

void write(uint16_t addr, uint16_t value) {
    Wire.beginTransmission(MLX90640_ADDR);
    Wire.write(addr >> 8);
    Wire.write(addr & 0xff);
    Wire.write(value >> 8);
    Wire.write(value & 0xff);
    Wire.endTransmission();
}

void readMany(uint16_t addr, int words) {
    while (words > 0) {
        Wire.beginTransmission(MLX90640_ADDR);
        Wire.write(addr >> 8);
        Wire.write(addr & 0xff);
        Wire.endTransmission(false);
        int n = min(16, words);
        Wire.requestFrom(MLX90640_ADDR, n * 2);
        for (int i = 0; i < n; i++) {
            Serial.write(Wire.read());
            Serial.write(Wire.read());
        }
        addr += n;
        words -= n;
    }
}

void readPage(int chess, int page) {
    uint16_t addr = 0x0400;
    for (int y = 0; y < 24; y++) {
        for (int h = 0; h < 2; h++) {
            Wire.beginTransmission(MLX90640_ADDR);
            Wire.write(addr >> 8);
            Wire.write(addr & 0xff);
            Wire.endTransmission(false);
            Wire.requestFrom(MLX90640_ADDR, 32);
            for (int i = 0; i < 16; i++) {
                int x = h * 16 + i;
                int p = chess ? (x ^ y) & 1 : y & 1;
                uint8_t msb = Wire.read();
                uint8_t lsb = Wire.read();
                if (p == page) {
                    Serial.write(msb);
                    Serial.write(lsb);
                }
            }
            addr += 16;
        }
    }
}

uint16_t serialRead2() {
    while (Serial.available() < 2);
    return Serial.read() << 8 | Serial.read();
}

void setup() {
    Wire.begin();
    Wire.setClock(400000);
    Serial.begin(115200);
}

void loop() {
    while (!Serial.available());
    uint8_t cmd = Serial.read();
    if (cmd == CMD_READ) {
        uint16_t addr = serialRead2();
        read(addr);
    } else if (cmd == CMD_WRITE) {
        uint16_t addr = serialRead2();
        uint16_t value = serialRead2();
        write(addr, value);
    } else if (cmd == CMD_READ_MANY) {
        uint16_t addr = serialRead2();
        uint16_t words = serialRead2();
        readMany(addr, words);
    } else if (cmd == CMD_READ_CHESS_PAGE) {
        uint8_t page = Serial.read();
        readPage(1, page);
    } else if (cmd == CMD_READ_INTERLEAVED_PAGE) {
        uint8_t page = Serial.read();
        readPage(0, page);
    }
}
