#include <assert.h>

int main() {
  _Bool v;
  char *ptr = &v;
  unsigned N = 0;
  ptr[N] = 0;
  assert(!v);
  return 0;
}

