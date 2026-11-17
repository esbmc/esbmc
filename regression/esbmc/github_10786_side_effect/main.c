// #10786: C17 6.5.3.4p2 evaluates the size of a VLA operand of sizeof.
#include <assert.h>

unsigned char nondet_uchar(void);

int main()
{
  int n = 1;
  sizeof(int[0 * n++]);
  sizeof(int[3][0 * n++]);
  assert(n == 3);

  unsigned char m = nondet_uchar();
  __ESBMC_assume(m > 0 && m < 10);
  assert(sizeof(int[m][3]) == 12u * m);
  assert(sizeof(char[3][m]) == 3u * m);
  return 0;
}
