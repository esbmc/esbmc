#include <assert.h>

typedef struct {
  int data;
  double d;
} s1;

typedef union {
  struct {
    int
      p0 : 1,
      p1 : 31;
  };
  int raw;
  double t;
} s2;

int main() {
  s1 x;

  int var = ((s2)(&x)->d).p1;
  int ta, tb;
  ta = var;
  var = 0;
  tb = var;
  assert(ta == tb);

  return 0;
}
