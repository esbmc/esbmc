#include <assert.h>

_Thread_local int counter = 5;
volatile int flag = 1;

int bump(void) {
  static _Thread_local int hits = 10;
  hits++;
  return hits;
}

int main() {
  counter += 2;
  assert(counter == 7);
  int a = flag;
  assert(a == 1);
  assert(bump() == 11);
  assert(bump() == 12);
  return 0;
}
