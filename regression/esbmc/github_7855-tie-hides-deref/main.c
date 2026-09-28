// esbmc/esbmc#7855: the pointer stored into the allocation is the address of a
// member of a null container, so reading it back and following it dereferences
// null. #7895 tied every flattened pointer to every other one sharing an
// address, which discarded that model and reported a proof. Reduced from
// SV-COMP's ldv-linux-4.0-rc1-mav dvb-ttusb-budget task, whose verdict flipped.

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

size_t nondet_size(void);

static void follow(struct node *stored)
{
  struct node copy = *stored;
  (void)copy.next->next;
}

int main(void)
{
  void *storage = malloc(nondet_size());
  alias = storage;
  alias->next = &container->node;
  follow(storage);
}
