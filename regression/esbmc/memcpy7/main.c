#include <string.h>
#include <assert.h>

struct A { int x; char y; };

int main() {
  struct A s1 = {42, 'a'};
  struct A s2;
  memcpy(&s2, &s1, sizeof(struct A));
  assert(s2.x == 42 && s2.y == 'a');
  return 0;
}
