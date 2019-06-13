#include <Time.h>
#include <TimeLib.h>


/**************************************************************************/
/*!
Adapted from Adafruit demo of Adafruit MCP9808 breakout
----> http://www.adafruit.com/products/1782
*/
/**************************************************************************/

#include <Wire.h>
#include "Adafruit_MCP9808.h"
const int lightPin = 0;
int sensorValue = 0;
Adafruit_MCP9808 tempsensor = Adafruit_MCP9808();

void setup() {
  Serial.begin(9600);
  while (!Serial); //waits for serial terminal to be open, necessary in newer arduino boards.
  pinMode(lightPin, INPUT);
  if (!tempsensor.begin(0x18)) {
    Serial.println("Couldn't find MCP9808! Check your connections and verify the address is correct.");
    while (1);
  }
    
  tempsensor.setResolution(3); // sets the resolution mode of reading, the modes are defined in the table bellow:

}

void loop() {
  tempsensor.wake();   // wake up, ready to read!
  float c = tempsensor.readTempC();
  float f = tempsensor.readTempF(); 
  Serial.print(c, 4); Serial.print(",");//C 
  Serial.print(f, 4); Serial.print(",");//F
  Serial.print(analogRead(lightPin)); //Serial.print(" Lux"); 
  delay(2000);
  tempsensor.shutdown_wake(1); // shutdown MSP9808 - power consumption ~0.1 mikro Ampere, stops temperature sampling
  Serial.println("");
  delay(200);
}
