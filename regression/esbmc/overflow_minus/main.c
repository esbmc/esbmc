#include <stdlib.h>

void assume_abort_if_not(int cond) {
  if(!cond) {abort();}
}

int minus(int a, int b) {
  assume_abort_if_not(b <= 0 || a >= b - 2147483648);
  assume_abort_if_not(b >= 0 || a <= b + 2147483647);
  return a - b;
}

int main()
{
  int a, b;
  a = nondet_int();
  b = nondet_int();
  minus(a, b);
  return 0;
}
