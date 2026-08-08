#include <stdint.h>
#include <assert.h>

unsigned long nondet_ulong(void);

char a[64];
char c = 0;

int main(void)
{
  // Widening the split under --deref-unknown-objects must only add cases, never
  // delete a property: this pointer names no object at all, so `invalid pointer'
  // has to survive the widening (esbmc/esbmc#6804).
  unsigned long u = nondet_ulong();
  char *p = (char *)(u * 8 - (uintptr_t)&a[0]);
  if ((uintptr_t)p < 64)
  {
    *p = 3;
    assert(c == 0);
  }
  return 0;
}
