// esbmc/esbmc#7965: a pointer into a live container, stored in malloc'd storage
// and read back through void *, is the pointer that was stored.
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

struct node *alias;
struct container live;
struct container *container = &live;

int main(void)
{
  void *storage = malloc(8);
  if (!storage)
    return 0;
  alias = storage;
  alias->next = &container->node;
  struct node copy = *(struct node *)storage;
  assert(copy.next == &live.node);
  (void)copy.next->next;
}
