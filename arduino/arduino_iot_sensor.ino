/*
 * PROJE: Enerji Tüketimi Tahmini İçin Veri Toplama Sistemi
 * DONANIM: Arduino Uno, HC-05, DHT11, LDR (+10k Direnç), Su Sensörü
 */

#include <SoftwareSerial.h>
#include <DHT.h>

// --- 1. PIN TANIMLAMALARI ---
#define RX_PIN 2       // HC-05'in TX bacağı buraya takılacak
#define TX_PIN 3       // HC-05'in RX bacağı buraya takılacak
#define DHT_PIN 4      // DHT11 Data bacağı
#define LDR_PIN A0     // LDR + Direnç birleşim noktası
#define WATER_PIN A1   // Su sensörü Sinyal ucu
#define LED_PIN 13     // Dahili LED

// --- 2. AYARLAR ve NESNELER ---
SoftwareSerial bluetooth(RX_PIN, TX_PIN);
#define DHTTYPE DHT11
DHT dht(DHT_PIN, DHTTYPE);

// YENİ: Hava durumu eşik değerleri
const int LDR_THRESHOLD_CLEAR = 700;
const int LDR_THRESHOLD_CLOUDY = 300;
const int WATER_THRESHOLD = 500;

void setup() {
  Serial.begin(9600);
  bluetooth.begin(9600);
  dht.begin();
  
  pinMode(LDR_PIN, INPUT);
  pinMode(WATER_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);  
  digitalWrite(LED_PIN, LOW); 

  Serial.println("=================================");
  Serial.println("IoT Energy System Started!");
  Serial.println("=================================");
  delay(1000);
}

void loop() {
  // --- A. VERİLERİ OKU ---
  float nem = dht.readHumidity();
  float sicaklik = dht.readTemperature();
  int isikDegeri = analogRead(LDR_PIN);
  int suDegeri = analogRead(WATER_PIN);

  // --- B. HATA KONTROLÜ ---
  if (isnan(nem) || isnan(sicaklik)) {
    Serial.println("HATA: DHT sensoru okunamadi! Kablolari kontrol et.");
    bluetooth.println("{\"error\":\"DHT_FAIL\"}"); 
    delay(2000);
    return;
  }

  // HAVA DURUMU CLUSTER HESAPLA
  int weatherCluster = hesaplaHavaDurumu(sicaklik, nem, isikDegeri, suDegeri);

  // JSON VERİ PAKETİ OLUŞTUR (CSV yerine)
  String jsonData = "{";
  jsonData += "\"temp\":" + String(sicaklik, 1) + ",";
  jsonData += "\"humidity\":" + String(nem, 0) + ",";
  jsonData += "\"light\":" + String(isikDegeri) + ",";
  jsonData += "\"water\":" + String(suDegeri) + ",";
  jsonData += "\"weather\":" + String(weatherCluster) + ",";
  jsonData += "\"holiday\":0";
  jsonData += "}";

  // --- E. VERİ GÖNDERME ---
  bluetooth.println(jsonData);
  
  Serial.print("Gonderilen Veri: ");
  Serial.println(jsonData);
  Serial.print("Hava Durumu: ");
  Serial.println(getWeatherName(weatherCluster)); 

  // F. PC'DEN KOMUT BEKLE
  if (bluetooth.available()) {
    String komut = bluetooth.readStringUntil('\n');
    komut.trim();
    
    Serial.print("Gelen Komut: ");
    Serial.println(komut);
    
    if (komut.startsWith("PREDICTION:")) {
      float tahmin = komut.substring(11).toFloat();
      
      Serial.print("Enerji Tahmini: ");
      Serial.print(tahmin);
      Serial.println(" kWh");
      
      // LED Kontrolü
      if (tahmin > 12.0) {
        digitalWrite(LED_PIN, HIGH);
        Serial.println(">>> YUKSEK TUKETIM - LED YANDI!");
      } else {
        digitalWrite(LED_PIN, LOW);
        Serial.println(">>> Normal tuketim - LED sondu");
      }
      
      bluetooth.println("OK");
    }
  }

  delay(3000); // 3 saniye (daha az spam)
}

// FONKSİYONLAR
int hesaplaHavaDurumu(float temp, float hum, int light, int water) {
  if (water > WATER_THRESHOLD) return 3; // Rainy
  if (temp < 10 && hum > 70) return 4;   // Cold/Snowy
  if (temp > 25 && light > LDR_THRESHOLD_CLEAR) return 0; // Clear
  if (light < LDR_THRESHOLD_CLOUDY) return 2; // Cloudy
  return 1; // Partly Cloudy
}

String getWeatherName(int cluster) {
  switch(cluster) {
    case 0: return "Clear/Sunny";
    case 1: return "Partly Cloudy";
    case 2: return "Cloudy";
    case 3: return "Rainy";
    case 4: return "Cold/Snowy";
    default: return "Unknown";
  }
}