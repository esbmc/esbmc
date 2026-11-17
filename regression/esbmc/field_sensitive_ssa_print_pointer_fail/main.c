#include <assert.h>
#include <stdlib.h>

struct node
{
  struct node *next;
};

struct container
{
  int header;
  struct node node;
};

struct container *container;

int main(void)
{
  void *storage = malloc(8);
  if (!storage)
    return 0;
  *(struct node **)storage = nondet_int() ? &container->node : 0;
  struct node copy = *(struct node *)storage;
  assert(copy.next == 0);
  return 0;
}
