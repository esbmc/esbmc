// A pointer computed from one rebuilt from bytes, stored and read back, is the
// pointer stored.
#include <assert.h>
#include <stdlib.h>
#include <string.h>

struct node { struct node *next; };
struct container { int header; struct node node; };
struct container *container;

int main(void)
{
  unsigned char *a = malloc(8), *b = malloc(8);
  if (!a || !b)
    return 0;
  struct node *p = &container->node, *q, *s;
  memcpy(a, &p, 8);
  memcpy(&q, a, 8);
  struct node *r = q + 1;
  *(struct node **)b = r;
  s = *(struct node **)b;
  assert(s == q + 1);
  return 0;
}
