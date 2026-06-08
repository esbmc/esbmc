#include <stdio.h>

extern int __VERIFIER_nondet_int(void);

// G-A soundness test. The format string is not a compile-time constant, so the
// formatted-output length (the sprintf return) cannot be bounded. ESBMC must
// model the return as an unconstrained non-negative int; adding it to a value
// near INT_MAX then has a reachable signed overflow. Before the printf
// soundness fix, a non-constant format made the modelled return 0, so this
// overflow was silently missed (false VERIFICATION SUCCESSFUL).
int main(void)
{
  char buf[64];
  int x = __VERIFIER_nondet_int();
  const char *fmt = (x & 1) ? "%d" : "%i"; // non-constant format
  int n = sprintf(buf, fmt, x);
  int base = 2147483640; // INT_MAX - 7
  int t = base + n;      // reachable overflow when n >= 8
  return t;
}
