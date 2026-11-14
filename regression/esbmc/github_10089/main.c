/* #10089: under --ir only a store that spans struct members fails; one that
 * stays inside a member at a dynamic offset does not. */
#include <assert.h>
struct S
{
  unsigned short a;
  unsigned short b;
  unsigned int c;
};
struct S s[2];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  int k = nondet_int();
  __ESBMC_assume(k == 1);
  unsigned int *p = (unsigned int *)&s[i] + k;
  *p = 7u;
  assert(s[i].c == 7u);
  return 0;
}
