// esbmc/esbmc#7965: a pointer into a NULL container, stored in malloc'd storage
// and read back through void *, must still be NULL-based when followed.
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
struct container *container;

int main(void)
{
  void *storage = malloc(8);
  alias = storage;
  alias->next = &container->node;
  struct node copy = *(struct node *)storage;
  (void)copy.next->next;
  assert(0);
}
