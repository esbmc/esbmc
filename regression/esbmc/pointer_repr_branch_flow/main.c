// A NULL-based pointer assigned under a branch that reads byte storage, then
// stored through untyped storage, reads back as the pointer stored.
#include <assert.h>
#include <stdlib.h>

struct node { struct node *next; };
struct container { int header; struct node node; };

struct node *alias;
struct container *container;
int g;

int main(void)
{
  void *storage = malloc(8);
  void *tmp = malloc(8);
  if (!storage || !tmp)
    return 0;
  *(int **)tmp = &g;
  struct node *n = 0;
  if (*(int **)tmp != 0)
    n = &container->node;
  alias = storage;
  alias->next = n;
  struct node copy = *(struct node *)storage;
  assert(copy.next == n);
  return 0;
}
