/* A pointee wider than the havoc covers must not keep its pre-loop value
 * past a loop invariant: the loop is left to the unwinder instead. */
#include <assert.h>

struct big
{
  int a[64];
};

struct big b;

int main()
{
  struct big *p = &b;
  int i = 0;
  __ESBMC_loop_invariant(i >= 0 && i <= 10);
  while (i < 10)
  {
    p->a[0] = 5;
    i++;
  }
  assert(b.a[0] == 5);
  return 0;
}
