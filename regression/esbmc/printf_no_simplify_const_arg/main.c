#include <assert.h>
#include <stdio.h>

int main()
{
  /* %02X of 0xAB is "AB" -- two characters -- whether the operand reaches the
     formatter as a cast, as a propagated symbol, or as a bare literal. */
  int a = printf("%02X", (unsigned)0xAB);
  assert(a == 2);

  unsigned v = 0xAB;
  int b = printf("%02X", v);
  assert(b == 2);

  int c = printf("%02X", 0xAB);
  assert(c == 2);
}
