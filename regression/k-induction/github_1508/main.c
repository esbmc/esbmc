#include <assert.h>

int main() {
  for (;;) {
    for (int i=0;
         i<2;
         i++)
      ;
    assert(-1 > 0);
  }
  return 0;
}
