#include <assert.h>

struct node {
  int value;
  struct node *next;
};

int main() {
  struct node b = {2, 0};
  struct node a = {1, &b};
  assert(a.next->value == 3);
  return 0;
}
