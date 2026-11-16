#include <assert.h>
#include <stddef.h>

extern void __VERIFIER_nondet_memory(void *ptr, size_t size);

int main()
{
  int a[2];
  __VERIFIER_nondet_memory(a, sizeof(a));
  if (a[0] == 0x01020304 && a[1] == -1)
    assert(0);
  return 0;
}
