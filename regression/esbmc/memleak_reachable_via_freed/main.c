// The global still points at the freed node, but a pointer stored in freed
// memory reaches nothing, so the buffer it held is leaked.
#include <stdlib.h>

struct node
{
  char *buf;
};

struct node *global;

int main()
{
  global = malloc(sizeof(struct node));
  if (!global)
    return 0;
  global->buf = malloc(8);
  free(global);
  return 0;
}
