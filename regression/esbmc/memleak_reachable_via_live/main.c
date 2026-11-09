// The buffer stays reachable through the live node the global points at.
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
  return 0;
}
