// The counterpart to github_7855-two-slots. Both pointers are flattened, so
// the tie between them is in play; it equates two flattened pointers only when
// they share an address, and these need not. An over-strong tie that merged
// every flattened pointer would prove this instead.
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

  const void *first, *second;
  slots[1].key = first;
  slots[2].key = second;

  __ESBMC_assert(slots[1].key == slots[2].key, "two slots read back equal");
  return 0;
}
