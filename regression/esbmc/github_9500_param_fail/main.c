#include <assert.h>

void rec(int n) {
  if (n == 0)
    return;
  rec(n - 1);
  assert(n != 2);
}

int main(void) {
  rec(3);
  return 0;
}
