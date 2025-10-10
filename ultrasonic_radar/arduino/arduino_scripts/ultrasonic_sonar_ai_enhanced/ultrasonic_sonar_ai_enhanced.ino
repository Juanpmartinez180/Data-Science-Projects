#include <stdio.h>

// --- Variables Globales ---
volatile int bufn, obufn;
const uint16_t bufsize = 2048;
const uint8_t bufnumber = 4;
uint16_t buf[bufnumber][bufsize]; // Búfer global (usado en segundo plano por el ADC)

// --- Definiciones de Pines ---
const int mux_chnl_sel_0 = 22;
const int mux_chnl_sel_1 = 24;
const int mux_chnl_sel_2 = 26;
const int mux_chnl_sel_3 = 28;

const int trigger_1 = 36;
const int trigger_2 = 38;
const int trigger_3 = 40;

// --- Configuración Inicial ---
void setup() {
  Serial.begin(250000);

  pinMode(mux_chnl_sel_0, OUTPUT);
  pinMode(mux_chnl_sel_1, OUTPUT);
  pinMode(mux_chnl_sel_2, OUTPUT);
  pinMode(mux_chnl_sel_3, OUTPUT);

  pinMode(trigger_1, OUTPUT);
  pinMode(trigger_2, OUTPUT);
  pinMode(trigger_3, OUTPUT);

  pmc_enable_periph_clk(ID_ADC);
  adc_setup();
  tc_adc_setup();
}

// --- Bucle Principal ---
void loop() {
  for (int j = 0; j < 3; j++) {

    if (j == 0) { // Sensor 1
      digitalWrite(mux_chnl_sel_0, LOW);
      digitalWrite(mux_chnl_sel_1, LOW);
      digitalWrite(trigger_1, HIGH);
      delayMicroseconds(50);
      digitalWrite(trigger_1, LOW);
    } else if (j == 1) { // Sensor 2
      digitalWrite(mux_chnl_sel_0, HIGH);
      digitalWrite(mux_chnl_sel_1, LOW);
      digitalWrite(trigger_2, HIGH);
      delayMicroseconds(50);
      digitalWrite(trigger_2, LOW);
    } else if (j == 2) { // Sensor 3
      digitalWrite(mux_chnl_sel_0, LOW);
      digitalWrite(mux_chnl_sel_1, HIGH);
      digitalWrite(trigger_3, HIGH);
      delayMicroseconds(50);
      digitalWrite(trigger_3, LOW);
    }

    delayMicroseconds(500);

    ADC->ADC_PTCR = ADC_PTCR_RXTDIS;

    // "Error B": Se redefine un búfer local. El DMA será redirigido aquí.
    uint16_t buf[bufnumber][bufsize];

    ADC->ADC_RPR = (uint32_t)buf[0];
    ADC->ADC_RCR = bufsize;
    ADC->ADC_RNPR = (uint32_t)buf[1];
    ADC->ADC_RNCR = bufsize;
    bufn = 1;
    obufn = 0;
    ADC->ADC_PTCR = ADC_PTCR_RXTEN;

    while (((obufn + 1) & 3) == bufn);

    // Enviar encabezado para Python
    Serial.print("--- Datos del Sensor ");
    Serial.print(j + 1);
    Serial.println(" ---");

    // Imprimir datos desde el búfer LOCAL
    for (int i = 0; i < bufsize; i++) {
      //float voltage = buf[obufn][i] * 3.3 / 4095.0;
      //Serial.println(voltage, 4);
      // Enviamos los 2 bytes crudos del valor uint16_t del ADC
      Serial.write((uint8_t*)&buf[obufn][i], sizeof(uint16_t));
    }
    Serial.println("--- Fin de Datos ---");

    obufn = (obufn + 1) & 3;
    
    delay(200);
  }
}

// --- Funciones de Configuración de Bajo Nivel ---
// NOTA: adc_setup() incluye el "Error A" para recrear la condición original.
void adc_setup() {
  ADC->ADC_CR = ADC_CR_SWRST;
  ADC->ADC_MR |= ADC_MR_TRGEN_EN | ADC_MR_TRGSEL_ADC_TRIG3 | ADC_MR_PRESCAL(0);
  ADC->ADC_IER = ADC_IER_ENDRX;
  ADC->ADC_CHER = ADC_CHER_CH7;
  NVIC_EnableIRQ(ADC_IRQn);
  
  // Apuntar el DMA al búfer GLOBAL inicialmente
  ADC->ADC_RPR = (uint32_t)buf[0];
  ADC->ADC_RCR = bufsize;
  ADC->ADC_RNPR = (uint32_t)buf[1];
  ADC->ADC_RNCR = bufsize;
  
  ADC->ADC_PTCR = ADC_PTCR_RXTEN;
  
  // "Error A": Inicia el ADC de forma prematura y continua
  ADC->ADC_CR = 2; 
}

void ADC_Handler() {
  if (ADC->ADC_ISR & ADC_ISR_ENDRX) {
    bufn = (bufn + 1) & 3;
    // La ISR sigue trabajando con el búfer GLOBAL, pero el loop lo ignora
    // y redirige el DMA a su búfer local.
    ADC->ADC_RNPR = (uint32_t)buf[bufn];
    ADC->ADC_RNCR = bufsize;
  }
}

void tc_adc_setup() {
  pmc_enable_periph_clk(ID_TC2);
  TC0->TC_CHANNEL[2].TC_CMR = TC_CMR_TCCLKS_TIMER_CLOCK2 | TC_CMR_WAVE | TC_CMR_WAVSEL_UP_RC | TC_CMR_ACPA_CLEAR | TC_CMR_ACPC_SET;
  TC0->TC_CHANNEL[2].TC_RC = 75;
  TC0->TC_CHANNEL[2].TC_RA = 38;
  TC0->TC_CHANNEL[2].TC_CCR = TC_CCR_SWTRG | TC_CCR_CLKEN;
}