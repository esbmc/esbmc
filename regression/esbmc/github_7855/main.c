// esbmc/esbmc#7855: an arbitrary pointer stored into a flexible array member
// of a malloc'd object is not the one read back. The store is lowered to a
// byte decomposition through the pointer's numeric address, and that address
// does not reconstruct the same pointer.
#include <stdlib.h>

struct entry
{
  const void *key;
  unsigned long hash_code;
};

struct state
{
  unsigned long mask;
  struct entry slots[];
};

int main(void)
{
  struct state *s = malloc(sizeof(struct state) + 4 * sizeof(struct entry));
  if (!s)
    return 0;

  const void *key;
  struct entry e;
  e.key = key;
  e.hash_code = 7;
  s->slots[1] = e;

  __ESBMC_assert(s->slots[1].key == key, "the stored key is the one read back");
  return 0;
}
