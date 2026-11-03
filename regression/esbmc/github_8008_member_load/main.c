// #8008
#include <assert.h>
struct node { int v; struct node *next; };
int main()
{
  struct node n2 = {1, 0};
  struct node n1 = {0, &n2};
  struct node *h = &n1;
  struct node *q = h->next;
  int a = n2.v + 1;
  q->v = 5;
  int b = n2.v + 1;
  assert(b == 2);
  return 0;
}
