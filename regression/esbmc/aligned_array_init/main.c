#include <assert.h>
#include <string.h>
int a[8] __attribute__((aligned(16))) = { 10, 20, 30, 30, 20, 10, 10, 20 };
int main() {
  int b[8];
  memcpy(b, a, sizeof(a));
  assert(b[0] == 10);
  assert(a[3] == 30);
  return 0;
}
