#include <string.h>
#include <assert.h>

struct Point { int x,y; };

int main() {
  struct Point pts1[2] = {{1,2},{3,4}};
  struct Point pts2[2];
  memcpy(pts2, pts1, sizeof(pts1));
  assert(pts2[1].x == 3 && pts2[1].y == 4);
  return 0;
}
