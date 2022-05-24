#include <assert.h>

typedef struct {
  int data;
} s1;

typedef union {
  struct {
    int
      p0 : 1,
      p1 : 31;
  };
  int raw;
} s2;

int main() {
  s1 x;
  if (((s2)(&x)->data).p1) {}

  int var, ta, tb;
  ta = var;
  var = 0;
  tb = var;
  assert(ta == tb);

  return 0;
}
