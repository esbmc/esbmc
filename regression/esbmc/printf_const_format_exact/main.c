#include <stdio.h>

// Precision-preservation control for the printf soundness fix. The format is a
// compile-time constant with no conversion specifiers, so the sprintf return is
// pinned to the exact length (5). The subsequent addition cannot overflow, so
// verification must still succeed — the fix must not over-approximate the
// fully-determined case.
int main(void)
{
  char buf[64];
  int n = sprintf(buf, "hello"); // exact length 5
  int base = 10;
  int t = base + n; // 15, no overflow
  return t;
}
