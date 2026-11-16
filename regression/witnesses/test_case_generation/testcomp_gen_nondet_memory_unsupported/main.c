#include <assert.h>
#include <stddef.h>

extern void __VERIFIER_nondet_memory(void *ptr, size_t size);
extern int __VERIFIER_nondet_int(void);

struct cplx
{
  _Complex double z;
  int i;
};

int main()
{
  /* Neither object has a byte layout the test generator knows, so their
   * bytes are not emitted and the test case is incomplete (only k);
   * generating it must still not abort. */
  int a[2][3];
  struct cplx c;
  __VERIFIER_nondet_memory(a, sizeof(a));
  __VERIFIER_nondet_memory(&c, sizeof(c));
  int k = __VERIFIER_nondet_int();
  if (a[1][2] == 5 && c.i == 7 && k == 9)
    assert(0);
  return 0;
}
