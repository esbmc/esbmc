#include <assert.h>
#include <stddef.h>

extern void __VERIFIER_nondet_memory(void *ptr, size_t size);

struct mixed
{
  float f;
  int *p[2];
  int v;
};

int main()
{
  /* f is laid out by its IEEE bits and the pointers as null; the harness
   * cannot reproduce an address. The padding bytes after f and v are free. */
  struct mixed m;
  __VERIFIER_nondet_memory(&m, sizeof(m));
  if (m.f == 1.5f && m.v == 5)
    assert(0);
  return 0;
}
