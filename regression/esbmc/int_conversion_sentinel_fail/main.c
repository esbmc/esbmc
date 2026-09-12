/* Negative counterpart of int_conversion_sentinel: the same source has to be
 * parsed and then actually verified, so a false assertion over the sentinel
 * struct must be refuted rather than swallowed by a parse failure. */
#include <assert.h>

struct lock
{
  void *owner;
  unsigned magic;
};

static struct lock l = {0xffffffffffffffffUL, 0xdead4ead};

int main(void)
{
  assert(l.magic == 0u);
  return 0;
}
