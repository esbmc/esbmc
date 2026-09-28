#include <stdint.h>
#include <assert.h>

unsigned long nondet_ulong(void);

char a[64], b[64];

int main(void)
{
  // A multiplication leaves the value set holding nothing but `unknown`, and
  // the subtraction must not be allowed to recover `a` from the other operand:
  // on this path the pointer is exactly &b[0], so the write clobbers b.
  // Treating a top value set as the integer side of the arithmetic would drop
  // the invalid-pointer property and prove this assertion (#6545).
  unsigned long u = nondet_ulong();
  char *p = (char *)(u * 8 - (uintptr_t)&a[0]);
  if ((uintptr_t)p == (uintptr_t)&b[0])
  {
    *p = 3;
    assert(b[0] == 0);
  }
  return 0;
}
