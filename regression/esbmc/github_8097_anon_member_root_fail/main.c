/* #8097: the value-set suffix of g.list.next runs through the anonymous union
 * member, named "node::(anonymous at main.c:...)", whose name holds a '.'. */
#include <stdlib.h>

struct node
{
  int kind;
  union
  {
    struct
    {
      struct node *next;
    } list;
    int spins;
  };
};

struct node g;

int main()
{
  g.list.next = malloc(sizeof(struct node));
  g.list.next = malloc(sizeof(struct node));
  return 0;
}
