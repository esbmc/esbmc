/* An allocation above PTRDIFF_MAX must fail, as glibc >= 2.30 does: a larger
   object makes pointer subtraction overflow, and its offsets would collide
   with the negative-offset encoding. --force-malloc-success removes the
   ordinary may-fail outcome, so NULL here can only come from the cap. */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

int main(void)
{
  char *over = malloc((size_t)PTRDIFF_MAX + 1);
  assert(over == NULL);
  return 0;
}
