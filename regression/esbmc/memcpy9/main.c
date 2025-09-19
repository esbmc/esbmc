#include <string.h>
#include <assert.h>

union U { int x; char y[4]; };

int main() {
  union U u1, u2;
  u1.x = 0x41424344; // 'ABCD'
  memcpy(&u2, &u1, sizeof(union U));
  assert(u2.x == 0x41424344);
  return 0;
}
