#include <string.h>
#include <assert.h>

struct B { int x:4; int y:28; };

int main() {
  struct B b1 = {5, 10};
  struct B b2;
  memcpy(&b2, &b1, sizeof(struct B));
  assert(b2.x == 5 && b2.y == 10);
  return 0;
}

