#include <assert.h>
#include <stdio.h>

int main()
{
  int x = 0;
  // %p outputs "0x" + pointer_width/4 hex digits. On 64-bit that is 18 chars;
  // on 32-bit that is 10 chars. Require at least 10 either way, which is
  // strictly larger than the "0" (length 1) the previous model emitted.
  int r = printf("%p", (void *)&x);
  assert(r >= 10);
}
