#include <assert.h>

typedef struct {
  int data;
} s1;

typedef struct {
  int raw;
} s2;

int main() {
  s1 x;

  int var = (s2){(&x)->data}.raw;
  int ta, tb;
  ta = var;
  var = 0;
  tb = var;
  assert(ta == tb);

  return 0;
}
