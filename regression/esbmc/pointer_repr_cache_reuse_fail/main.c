// The same pointer stored into two byte-flattened objects reads back equal,
// so asserting it differs must fail.
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
  assert(t[1].key != key);
  return 0;
}
