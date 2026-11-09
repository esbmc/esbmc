/* #10089: under --ir a store at a dynamic offset spanning two struct members
 * was dropped, so the second assertion verified. */
#include <assert.h>
struct S
{
  unsigned int a;
  unsigned int b;
};
struct S s[2];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  unsigned char *p = (unsigned char *)&s[i];
  *(unsigned long long *)p = 0x0000000200000001ULL;
  assert(s[i].b == 0);
  return 0;
}
