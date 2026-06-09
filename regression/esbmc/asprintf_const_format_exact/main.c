#include <stdio.h>
#include <stdlib.h>

// Regression test: asprintf was not recognised as printf-family before G-D
// wiring. ESBMC now routes it through symex_printf and correctly models the
// return value for a constant format with no conversion specifiers (5 chars).
// The subsequent addition cannot overflow, so verification must succeed.
//
// Note: the symex_printf model does not allocate the output buffer pointed to
// by *strp — only the return length is modelled.  We avoid dereferencing or
// freeing the result pointer to stay within the modelled behaviour.
int main(void)
{
  char *s = NULL;
  int n = asprintf(&s, "hello"); // exact length 5
  if (n < 0)
    return 1; // allocation failure — no overflow possible
  int base = 10;
  int t = base + n; // 15, no overflow
  return t;
}
