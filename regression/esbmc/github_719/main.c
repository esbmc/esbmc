#include <assert.h>

typedef union {
  struct {};
} s1;

typedef struct {
  int x[2];
} s2;

s1 v1;
s2 v2;

int main() {
  for (int i = 0; i < 2; i++) {
    v2.x[i] = 0;
  }

  int x = 0;
  for (int i = 0; i < 2; i++) {
    x++;
  }
  assert(x == 0);

  return 0;
}
