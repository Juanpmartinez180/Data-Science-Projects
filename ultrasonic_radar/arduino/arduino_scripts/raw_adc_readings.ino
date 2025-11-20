#include<stdio.h>

#define BUFFER_LEN 2000

int raw_reading;
float buff[BUFFER_LEN];
int count = 0; //counter for number of measurements
int i; //aux var
int t = 0;
int elapsed = 0;
bool measurement_flg = false;

const int trigger = 36;
const int echo = 37;


void setup() {
  Serial.begin(115200);

  pinMode(trigger, OUTPUT);
  pinMode(echo, INPUT);
  
  ADC->ADC_CR = ADC_CR_SWRST; // Reset ADC
  ADC->ADC_MR |= 0x80; // ADC on free running mode
  ADC->ADC_WPMR = 0; //Disable write protection 
  ADC->ADC_CHER = ADC_CHER_CH7; //Select CH7 as ADC Channel register
  ADC->ADC_CR = 2; //Start conversion

}

void loop() {

  //trigger de echo
  digitalWrite(trigger, HIGH);
  delayMicroseconds(500);
  digitalWrite(trigger, LOW);

  delayMicroseconds(1100); //Delay to avoid internal echos
  measurement_flg = true; //Raise measurement flag.. initialize value reading

  while(measurement_flg){
    if(count >= BUFFER_LEN){
      for(i = 0; i < BUFFER_LEN; ++i){
        Serial.println(buff[i]);
      }
      count = 0;
      delay(1000); //wait 1sec until new measurement
      measurement_flg = false; //down measurement flag. Wait for new sent echo.
    }
    else{
      while((ADC->ADC_ISR & ADC_ISR_EOC7) == 0); //wait for the end of conversion
      raw_reading = ADC->ADC_CDR[7];    // read data, store into the buffer  
      buff[count] = raw_reading*3.3/4096.0;
  
      count += 1; //increment measurements counter by 1
    };
  };
  

}
