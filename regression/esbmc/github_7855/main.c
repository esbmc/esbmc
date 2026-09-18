// esbmc/esbmc#7855: a pointer stored into a malloc'd object ESBMC models as
// untyped bytes is not the one read back. A stack array of the same type
// verifies; see github_7855-cast for what the store is lowered to.
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

  const void *key;
  slots[1].key = key;

  __ESBMC_assert(slots[1].key == key, "the stored key is the one read back");
  return 0;
}
