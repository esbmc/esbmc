#include <assert.h>

struct node {
  int value;
  struct node *next;
};

struct b_t;
struct a_t { int x; struct b_t *b; };
struct b_t { int y; struct a_t *a; };

int main() {
  struct node b = {2, 0};
  struct node a = {1, &b};
  assert(a.value == 1);
  assert(a.next->value == 2);
  assert(a.next->next == 0);

  struct a_t ma = {1, 0};
  struct b_t mb = {2, &ma};
  ma.b = &mb;
  assert(ma.b->y == 2);
  assert(ma.b->a->x == 1);
  assert(mb.a->b == &mb);

  return 0;
}
