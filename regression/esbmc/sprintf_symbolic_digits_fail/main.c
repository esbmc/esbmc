#include <stdio.h>
int main() {
  unsigned char c = nondet_uchar();
  char b[8];
  sprintf(b, "%o", c);
  __ESBMC_assert(b[1] != 0, "false: c<8 prints one digit");
  return 0;
}
