#include <assert.h>
union {
  unsigned : 4;
  int a;
} b = {};

int main() {
  b.a = 1;
  assert(b.a == 2);
  return 0;
}
