#include <assert.h>

int main() {
  int x,y;
  // must fail
  assert((x-y>0) == (x>y));
}
