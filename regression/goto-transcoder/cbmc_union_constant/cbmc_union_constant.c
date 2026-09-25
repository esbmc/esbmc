#include <assert.h>

union mix {
  int i;
  float f;
};

union mix g = {.i = 11};

int main() {
  union mix u = {.i = 42};
  assert(u.i == 42);

  union mix v;
  v.f = 1.5f;
  assert(v.f == 1.5f);

  assert(g.i == 11);

  union mix w = v;
  assert(w.f == 1.5f);

  return 0;
}
