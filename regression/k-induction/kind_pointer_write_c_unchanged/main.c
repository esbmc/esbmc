#include <assert.h>
#include <stdlib.h>

struct list
{
  unsigned size;
  int *slot;
};

unsigned nondet_uint(void);

int main(void)
{
  struct list *l = malloc(sizeof(struct list));
  if (!l)
    return 0;
  l->size = 0;
  l->slot = malloc(16 * sizeof(int));
  if (!l->slot)
    return 0;
  unsigned n = nondet_uint();
  for (unsigned i = 0; i < n; i++)
  {
    l->slot[i % 16] = 1;
    l->size++;
  }
  assert(l->size == n);
  return 0;
}
