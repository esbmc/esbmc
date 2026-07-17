#include <assert.h>

union mix {
  int i;
  float f;
};

int main() {
  union mix u = {.i = 42};
  assert(u.i == 41);
  return 0;
}
