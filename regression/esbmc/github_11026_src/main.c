#include <assert.h>
#include <string.h>

unsigned nondet_uint(void);

int main()
{
  int a[16] = {1, 2, 3, 4, 5, 6, 7, 8}, b[16] = {0};
  unsigned s = nondet_uint(), m = nondet_uint();
  __ESBMC_assume(s <= 8 && m <= 16 - s);
  memmove(b, a + s, m * sizeof(int));
  unsigned j = nondet_uint();
  __ESBMC_assume(j < 16);
  assert(b[j] == (j < m ? a[s + j] : 0));
  return 0;
}
