int main() {
   float b = nondet_float();
   float c = (int)b;
   assert((long)c > 0); // Crashes with Boolector 
   assert((short)c > 0); // Crashes with Boolector 
   assert((char)c > 0); // Crashes with Boolector 
   assert((float)c > 0); // Works with Boolector
}
