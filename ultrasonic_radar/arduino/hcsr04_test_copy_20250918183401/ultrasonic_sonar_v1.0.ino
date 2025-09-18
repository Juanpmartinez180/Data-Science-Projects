#include<stdio.h>

volatile int bufn,obufn;       //status variables
const uint16_t bufsize = 2048;     //Lenght of each buffer       
const uint8_t bufnumber = 4;      //Number of buffers      

uint16_t buf[bufnumber][bufsize];  //buffer

const int mux_chnl_sel_0 = 22;  //Board pin number - Multiplexor channel selection S0
const int mux_chnl_sel_1 = 24;  //Board pin number - Multiplexor channel selection S1

const int trigger_1 = 32;    //Board pin number - Trigger of sensor #1
const int trigger_2 = 38;    //Board pin number - Trigger of sensor #2
const int trigger_3 = 40;    //Board pin number - Trigger of sensor #3


void setup() {
  
  Serial.begin(115200); // Initialize serial comm

  pmc_enable_periph_clk(ID_ADC);

  // Define board pin schema
  pinMode(mux_chnl_sel_0, OUTPUT);
  pinMode(mux_chnl_sel_1, OUTPUT);

  pinMode(trigger_1, OUTPUT);  
  pinMode(trigger_2, OUTPUT);
  pinMode(trigger_3, OUTPUT);

  // Select MUX channel C0 (S0 = 0, S1 = 0, S2 = 0, S3 = 0)
  digitalWrite(mux_chnl_sel_0, LOW);
  digitalWrite(mux_chnl_sel_1, LOW);

  // Initialize ADC and capture timer
  adc_setup();          //Initialize ADC setup
  tc_adc_setup();       //Initialize timer setuo
}

void loop() {
  int j= 0;

  for(j == 0; j < 3; j++){
    int i = 0;

    if(j==0){

       // Select MUX channel C0 (S0 = 0, S1 = 0, S2 = 0, S3 = 0)
      digitalWrite(mux_chnl_sel_0, LOW);
      digitalWrite(mux_chnl_sel_1, LOW);

      //trigger de echo
      digitalWrite(trigger_3, HIGH);
      delayMicroseconds(50);
      digitalWrite(trigger_3, LOW);
    
      delayMicroseconds(500);   //Delay before the capture to avoid internal echoes
      
      }
    else if(j==1){

      // Select MUX channel C1 (S0 = 1, S1 = 0, S2 = 0, S3 = 0)
      digitalWrite(mux_chnl_sel_0, HIGH);
      digitalWrite(mux_chnl_sel_1, LOW);

      //trigger de echo
      digitalWrite(trigger_2, HIGH);
      delayMicroseconds(50);
      digitalWrite(trigger_2, LOW);
    
      delayMicroseconds(500);   //Delay before the capture to avoid internal echoes
      
      }
    else if(j==2){
      
      // Select MUX channel C3 (S0 = 0, S1 = 1, S2 = 0, S3 = 0)
      digitalWrite(mux_chnl_sel_0, HIGH);
      digitalWrite(mux_chnl_sel_1, HIGH);

      //trigger de echo
      digitalWrite(trigger_1, HIGH);
      delayMicroseconds(50);
      digitalWrite(trigger_1, LOW);
    
      delayMicroseconds(500);   //Delay before the capture to avoid internal echoes
      
      }
  
    ADC->ADC_PTCR = ADC_PTCR_RXTDIS;  //Disable the DMA data transfer
    
    uint16_t buf[bufnumber][bufsize]; //Redefine buffers
  
    ADC->ADC_RPR = (uint32_t)buf[0];                      // DMA buffer - Receive pointer register
    ADC->ADC_RCR = bufsize;                               // Receive counter register
    ADC->ADC_RNPR = (uint32_t)buf[1];                     // next DMA buffer
    ADC->ADC_RNCR = bufsize;
    bufn = 1;
    obufn = 0;
  
    ADC->ADC_PTCR = ADC_PTCR_RXTEN;  //Enable the DMA data trasnfer
  
    while((obufn + 1)%4==bufn); // wait for buffer to be full
    
    for(i == 0; i < bufsize; i++){
      Serial.println(buf[obufn][i]*3.3/4095.0);
      delayMicroseconds(30);
    };
    
    obufn = (obufn+1)&3;  //Update old-buffer index
    delay(2000); //wait 1sec until new measurement
  };
  

}


/*************  Configure adc_setup function  *******************/
void adc_setup() {
  
    //ADC SETUP
  PMC->PMC_PCER1 |= PMC_PCER1_PID37;                    // ADC power ON
  
  ADC->ADC_CR = ADC_CR_SWRST;                           // Reset ADC
  ADC->ADC_MR |=  ADC_MR_TRGEN_EN                       // Hardware trigger select
                  | ADC_MR_TRGSEL_ADC_TRIG3             // Trigger by TIOA1
                  | ADC_MR_PRESCAL(0);
  ADC->ADC_IDR = ~ADC_IDR_ENDRX;                        //Interrupt disable register. End of receive buffer interrupt disable.
  ADC->ADC_IER = ADC_IER_ENDRX;                         // End Of Conversion interrupt enable for channel 7
  
  
  ADC->ADC_CHER = ADC_CHER_CH7;                         //Select CH7 as ADC Channel register - A0 Arduino board pin
  
  ADC->ADC_IDR=~(1<<27);
  ADC->ADC_IER=1<<27;
  NVIC_EnableIRQ(ADC_IRQn);                             // Enable ADC interrupt
  
    /*********  PDC/DMA  buffer filling sequence **********/  
  ADC->ADC_RPR = (uint32_t)buf[0];                      // DMA buffer - Receive pointer register
  ADC->ADC_RCR = bufsize;                               // Receive counter register
  ADC->ADC_RNPR = (uint32_t)buf[1];                     // next DMA buffer
  ADC->ADC_RNCR = bufsize;                              // next DMA buffer counter register
  
  bufn = 1;     //next buffer number filled
  obufn = 0;    //buffer number filled - 'old buff number'
  
  ADC->ADC_PTCR |= 1 ;            // Enable PDC Receiver channel request - Transfer control register. Receiver transfer enabled  ADC_PTCR_RXTEN
  ADC->ADC_CR = 2;                //ADC_CR_START
}


void ADC_Handler()
{   
  int f=ADC->ADC_ISR;       //Check ADC ISR - Interrupt status register 
  
  if (f&(1<<27)){            //If flag 27 (ENDRX - End of RX buffer) is raised
    bufn=(bufn+1)&3;           //Increase the buffer index by 1      bufn=(bufn+1)&3;  https://forum.arduino.cc/t/speed-of-analogread/134428/40#msg2526475  https://gist.github.com/pklaus/5921022
    ADC->ADC_RNPR=(uint32_t)buf[bufn];   //Asign the new buffer to be filled (address)
    ADC->ADC_RNCR=bufsize;                 //and size
  }
}


/*******  Timer Counter 0 Channel 2 to generate PWM pulses thru TIOA2 for the ADC********/
void tc_adc_setup() { 

  PMC->PMC_PCER0 |= PMC_PCER0_PID29;                      // TC1 power ON : Timer Counter 0 channel 2 IS TC1
  TC0->TC_WPMR = 0;
  TC0->TC_CHANNEL[2].TC_CMR = TC_CMR_TCCLKS_TIMER_CLOCK2  // MCK/8, clk on rising edge. MCK = 84MHZ
                              | TC_CMR_WAVE               // Waveform mode
                              | TC_CMR_WAVSEL_UP_RC       // UP mode with automatic trigger on RC Compare
                              | TC_CMR_ACPA_CLEAR         // Clear TIOA2 on RA compare match
                              | TC_CMR_ACPC_SET;          // Set TIOA2 on RC compare match

  TC0->TC_CHANNEL[2].TC_RC = 75;  //<*********************  Frequency = (Mck/8)/TC_RC
  TC0->TC_CHANNEL[2].TC_RA = 38;  //<********************   Any Duty cycle in between 1 and TC_RC

  TC0->TC_WPMR = 1;
  
  TC0->TC_CHANNEL[2].TC_CCR =  TC_CCR_SWTRG | TC_CCR_CLKEN;               // TC2 reset and enable

}
