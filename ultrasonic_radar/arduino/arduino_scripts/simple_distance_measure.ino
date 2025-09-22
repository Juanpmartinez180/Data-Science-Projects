const int trigger = 36;
const int echo = 37;
float distance1, distance2, echo_1, echo_2 ;

void setup() {
  Serial.begin(9600);
  pinMode(trigger, OUTPUT);
  pinMode(echo, INPUT);

}

void loop() {
  // Start measurement
  digitalWrite(trigger, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigger, LOW);

  // Acquire echo
  echo_1 = pulseIn(echo, HIGH);
  distance1 = (echo_1/2 ) / 29.1;

 

  // print the value on the serial monitor
  Serial.print("\nDistance1 ");
  Serial.print(distance1);
  Serial.print(" cm\n");
  delay(1000);
}
