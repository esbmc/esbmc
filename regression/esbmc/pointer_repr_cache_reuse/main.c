// The same pointer stored into two byte-flattened objects: the second store
// reuses the first store's conversion, and must still read back as stored.
#include <assert.h>
#include <stdlib.h>

void *nondet_ptr(void);

struct entry
{
  const void *key;
  unsigned long h;
};

int main(void)
{
  struct entry *s = malloc(4 * sizeof(struct entry));
  struct entry *t = malloc(4 * sizeof(struct entry));
  if (!s || !t)
    return 0;
  const void *key = nondet_ptr();
  s[1].key = key;
  t[1].key = key;
  assert(t[1].key == key);
  return 0;
}
