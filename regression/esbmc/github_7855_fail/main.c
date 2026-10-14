// esbmc/esbmc#7855: the counterpart to github_7855. The byte round trip returns
// the pointer that was stored and no other, so an unrelated pointer must still
// compare unequal -- an encoding that collapsed would pass this too.
#include <stdlib.h>

struct entry
{
  const void *key;
  unsigned long hash_code;
};

int main(void)
{
  struct entry *slots = malloc(4 * sizeof(struct entry));
  if (!slots)
    return 0;

  const void *key, *other;
  slots[1].key = key;

  __ESBMC_assert(slots[1].key == other, "an unrelated pointer reads back equal");
  return 0;
}
