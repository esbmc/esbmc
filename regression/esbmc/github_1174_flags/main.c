#include <assert.h>
#include <stdio.h>

int main()
{
  // %-5d| of 3: flag '-' is parsed, width 5 pads with spaces → "    3|"
  // (6 chars). Regardless of whether left/right justification is modelled,
  // the total length of a width-5 field plus '|' is 6.
  int r1 = printf("%-5d|", 3);
  assert(r1 == 6);

  // Width 5 on %d with value 3 → 5-char field.
  int r2 = printf("%5d", 3);
  assert(r2 == 5);

  // Flag '0' on %d with width 4 → "0003" (4 chars, zero-padded).
  int r3 = printf("%04d", 3);
  assert(r3 == 4);
}
