#include <assert.h>
#include <stddef.h>
struct S
{
  int a;
  int arr[4];
};
int nondet_int(void);
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 4);
  size_t o = offsetof(struct S, arr[i]);
  assert(o != 12);
  return 0;
}
