// esbmc/esbmc#7855: two pointers flattened into neighbouring slots each read
// back as themselves, so the byte round trip does not merge them.
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

  __ESBMC_assert(slots[1].key == first, "slot 1 reads back its own pointer");
  __ESBMC_assert(slots[2].key == second, "slot 2 reads back its own pointer");
  return 0;
}
