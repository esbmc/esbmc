// esbmc/esbmc#7855: flattening a pointer into a malloc'd object ESBMC models
// as untyped bytes and reading it back must yield the pointer that went in.
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
