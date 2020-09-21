long nondet_long(void);
 
int main(void) {
  long l = 0;
  int c = 0;
  long t = nondet_long();
  c += (l < t);
}

