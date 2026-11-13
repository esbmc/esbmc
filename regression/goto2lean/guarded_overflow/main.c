unsigned nondet_uint(void);
int main(void) {
  unsigned a = nondet_uint(), b = nondet_uint(), c = nondet_uint();
  unsigned r = c ? a + b : 0;
  _Bool t = c && a + b > 3;
  return 0;
}
