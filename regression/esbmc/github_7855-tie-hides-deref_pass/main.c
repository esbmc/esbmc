// The twin of github_7855-tie-hides-deref over the same construct: the member
// address flattened into the allocation reads back as itself, which is the
// round trip #7855 is about.
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

struct node *alias;
struct container *container;

int main(void)
{
  struct container live;
  container = &live;

  void *storage = malloc(sizeof(struct node));
  if (!storage)
    return 0;

  alias = storage;
  alias->next = &container->node;

  struct node copy = *(struct node *)storage;
  __ESBMC_assert(
    copy.next == &container->node, "the stored member address reads back");
  return 0;
}
