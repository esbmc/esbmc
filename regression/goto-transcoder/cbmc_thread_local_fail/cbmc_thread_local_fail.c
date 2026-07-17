#include <assert.h>

_Thread_local int counter = 5;

int main() {
  counter += 2;
  assert(counter == 8);
  return 0;
}
