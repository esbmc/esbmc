int nondet_int(void);
unsigned nondet_uint(void);
int main(void) {
  int x = nondet_int(), y = nondet_int();
  unsigned a = nondet_uint(), b = nondet_uint();
  int s = x - y;
  int p = x * y;
  int q = x / y;
  int r = x % y;
  int n = -x;
  unsigned us = a - b;
  unsigned um = a * b;
  unsigned un = -a;
  return 0;
}
