/*
 * PROJE: Enerji Tüketimi Tahmini İçin Veri Toplama Sistemi
 * DONANIM: Arduino Uno, HC-05, DHT11, LDR (+10k Direnç), Su Sensörü
 */

#include <SoftwareSerial.h>
#include <DHT.h>

// --- 1. PIN TANIMLAMALARI ---
#define RX_PIN 2       // HC-05'in TX bacağı
#define TX_PIN 3       // HC-05'in RX bacağı
#define DHT_PIN 4      // DHT11 Data bacağı
#define LDR_PIN A0     // LDR + Direnç
#define WATER_PIN A1   // Su sensörü

// --- 2. AYARLAR ---
SoftwareSerial bluetooth(RX_PIN, TX_PIN);
#define DHTTYPE DHT11
DHT dht(DHT_PIN, DHTTYPE);

// Hava durumu eşikleri
const int LDR_THRESHOLD_CLEAR = 700;
const int LDR_THRESHOLD_CLOUDY = 300;
const int WATER_THRESHOLD = 500;

// Tahmin değeri (PC'den gelecek)
float currentPrediction = 0.0;

void setup() {
  Serial.begin(9600);
  bluetooth.begin(9600);
  dht.begin();
  
  delay(2000);
  
  Serial.println("=================================");
  Serial.println("IoT Energy System Started!");
  Serial.println("=================================");
}

int hesaplaHavaDurumu(float temp, float hum, int light, int water) {
  if (water > WATER_THRESHOLD) return 3; // Rainy
  if (temp < 10 && hum > 70) return 4;   // Cold/Snowy
  if (temp > 25 && light > LDR_THRESHOLD_CLEAR) return 0; // Clear
  if (light < LDR_THRESHOLD_CLOUDY) return 2; // Cloudy
  return 1; // Partly Cloudy
}

void loop() {
  // 1. Sensörlerden veri oku
  float nem = dht.readHumidity();
  float sicaklik = dht.readTemperature();
  int isikDegeri = analogRead(LDR_PIN);
  int suDegeri = analogRead(WATER_PIN);
  
  // 2. Hata kontrolü
  if (isnan(nem) || isnan(sicaklik)) {
    Serial.println("HATA: DHT sensoru okunamadi!");
    bluetooth.println("{\"error\":\"DHT_FAIL\"}");
    delay(2000);
    return;
  }
  
  // 3. Hava durumu hesapla
  int weatherCluster = hesaplaHavaDurumu(sicaklik, nem, isikDegeri, suDegeri);
  
  // 4. JSON oluştur
  String jsonData = "{";
  jsonData += "\"temp\":" + String(sicaklik, 1) + ",";
  jsonData += "\"humidity\":" + String(nem, 0) + ",";
  jsonData += "\"light\":" + String(isikDegeri) + ",";
  jsonData += "\"water\":" + String(suDegeri) + ",";
  jsonData += "\"weather\":" + String(weatherCluster) + ",";
  jsonData += "\"holiday\":0";
  jsonData += "}";
  
  // 5. Gönder - SADECE JSON GÖNDERİYORUZ!
  Serial.println(jsonData);      // PC/Streamlit için
  bluetooth.println(jsonData);   // Bluetooth için
  
  // 6. Debug bilgisi (opsiyonel - Serial monitörde görünür)
  // Serial.print("Debug - Sicaklik: ");
  // Serial.println(sicaklik);
  
  // 7. PC'den tahmin al (opsiyonel)
  if (bluetooth.available()) {
    String komut = bluetooth.readStringUntil('\n');
    
    if (komut.startsWith("PREDICTION:")) {
      currentPrediction = komut.substring(11).toFloat();
    }
  }
  
  // Serial'dan da tahmin alabilir (USB bağlantısı için)
  if (Serial.available()) {
    String komut = Serial.readStringUntil('\n');
    
    if (komut.startsWith("PREDICTION:")) {
      currentPrediction = komut.substring(11).toFloat();
    }
  }
  
  delay(2000);  // 2 saniyede bir veri gönder
}