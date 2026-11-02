/* #9693: a wrapper's malloc(n) is typed by the caller's sizeof(T). */
#include <stdlib.h>
#include <assert.h>
struct node
{
  struct node *left, *right, *parent;
  int value;
};
void *safe_malloc(size_t size)
{
  void *p = malloc(size);
  if (!p)
    abort();
  return p;
}
int main(void)
{
  struct node *last = 0, *n = 0;
  for (int i = 0; i < 4; i++)
  {
    n = safe_malloc(sizeof(struct node));
    n->left = 0;
    n->right = last;
    n->parent = 0;
    n->value = i;
    if (last)
      last->parent = n;
    last = n;
  }
  while (n->right)
    n = n->right;
  assert(n->parent == 0 || n->parent->value == 2);
  return 0;
}
