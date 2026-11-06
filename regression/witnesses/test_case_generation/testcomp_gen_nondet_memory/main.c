#include <assert.h>
#include <stddef.h>

extern void __VERIFIER_nondet_memory(void *ptr, size_t size);
extern int __VERIFIER_nondet_int(void);

struct pair
{
  int x, y;
};

int main()
{
  /* ESBMC havocs the whole struct in one step; the test harness reads one
   * __VERIFIER_nondet_uchar() per byte, so the test case lists its 8 bytes
   * before k. */
  struct pair p = {0, 0};
  __VERIFIER_nondet_memory(&p, sizeof(p));
  int k = __VERIFIER_nondet_int();
  if (p.x == 0x01020304 && p.y == -1 && k == 9)
    assert(0);
  return 0;
}
