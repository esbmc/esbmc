/* An allocation above PTRDIFF_MAX must fail, as glibc >= 2.30 does: a larger
   object makes pointer subtraction overflow, and its offsets would collide
   with the negative-offset encoding. The byte below the cap must still
   succeed, so the boundary is pinned from both sides. --force-malloc-success
   removes the ordinary may-fail outcome, leaving only the cap. */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

int main(void)
{
  char *over = malloc((size_t)PTRDIFF_MAX + 1);
  assert(over == NULL);

  char *at = malloc((size_t)PTRDIFF_MAX);
  assert(at != NULL);

  return 0;
}
