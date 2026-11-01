#include <stdio.h>
#include <string.h>
int main() {
  unsigned char c = nondet_uchar();
  int v = nondet_int();
  char b[32];
  int n = sprintf(b, "\\%03o", c);
  __ESBMC_assert(b[0] == '\\' && b[1] == '0' + (c >> 6) && b[2] == '0' + ((c >> 3) & 7) && b[3] == '0' + (c & 7) && b[4] == 0 && n == 4, "oct");
  n = sprintf(b, "[%5d|%-4x|%04X]", v, c, c);
  if (v == -42) __ESBMC_assert(strcmp(b, "[  -42|") == 0 || (b[0]=='[' && b[1]==' ' && b[2]==' ' && b[3]=='-' && b[4]=='4' && b[5]=='2' && b[6]=='|'), "d");
  if (c == 0xab) __ESBMC_assert(b[n-6]=='|' && b[n-5]=='0' && b[n-4]=='0' && b[n-3]=='A' && b[n-2]=='B' && b[n-1]==']' && b[n]==0, "X");
  if (v == -2147483647 - 1) __ESBMC_assert(n == 1 + 11 + 1 + 4 + 1 + 4 + 1 && b[1] == '-' && b[2] == '2' && b[11] == '8', "min");
  sprintf(b, "%o", c);
  __ESBMC_assert(b[0] != '9', "never 9");
  return 0;
}
