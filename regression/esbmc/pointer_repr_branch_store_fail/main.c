// A pointer stored on either arm of a branch reads back equal on both arms,
// so asserting it differs must fail.
#include <assert.h>
#include <stdlib.h>

void *nondet_ptr(void);
int nondet_int(void);

struct entry
{
  const void *key;
  unsigned long h;
};

int main(void)
{
  struct entry *s = malloc(4 * sizeof(struct entry));
  if (!s)
    return 0;
  const void *key = nondet_ptr();
  int i = nondet_int() ? 1 : 2;
  if (i == 1)
    s[1].key = key;
  else
    s[2].key = key;
  assert(s[i].key != key);
  return 0;
}
