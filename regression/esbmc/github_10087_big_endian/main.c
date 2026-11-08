/* #10087: on big-endian targets the first bit-field holds the high bits. */
#include <assert.h>
struct S
{
  unsigned int a : 4;
  unsigned int b : 28;
  unsigned int c;
};
struct S s[2];
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  *(unsigned int *)&s[i] = 0x10000031u;
  assert(s[i].a == 1);
  assert(s[i].b == 0x31);
  assert(*(unsigned int *)&s[i] == 0x10000031u);
  return 0;
}
