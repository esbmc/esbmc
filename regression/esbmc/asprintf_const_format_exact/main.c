#include <stdio.h>
#include <stdlib.h>

// Regression test: asprintf was not recognised as printf-family before G-D
// wiring. ESBMC now routes it through symex_printf and correctly models the
// return value for a constant format with no conversion specifiers (5 chars).
// The subsequent addition cannot overflow, so verification must succeed.
//
// The symex_printf model also allocates a 1-byte tracked heap buffer for *strp
// (GitHub #5139/#5141 fix). We avoid dereferencing with exact-size checks
// since the buffer is modelled as 1 byte regardless of format length.
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
