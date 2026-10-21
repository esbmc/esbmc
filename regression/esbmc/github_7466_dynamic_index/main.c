#include <assert.h>
#include <stddef.h>
struct S
{
  int a;
  int arr[4];
  struct
  {
    char x;
    int y[3];
  } in[2];
};
int nondet_int(void);
int main(void)
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 4);
  size_t o = offsetof(struct S, arr[i]);
  assert(o == 4 + 4 * (size_t)i);
  int j = nondet_int();
  __ESBMC_assume(j >= 0 && j < 2);
  int k = nondet_int();
  __ESBMC_assume(k >= 0 && k < 3);
  size_t o2 = offsetof(struct S, in[j].y[k]);
  assert(o2 == 20 + 16 * (size_t)j + 4 + 4 * (size_t)k);
  return 0;
}
