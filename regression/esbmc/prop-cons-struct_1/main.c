#include <assert.h>

struct Point {
  int x;
  int y;
};

int main() {
  struct Point arr[2];
  arr[0].x = 10;
  arr[0].y = 20;
  arr[1].x = 5;
  arr[1].y = 15;

  // Expect propagation across array element fields
  assert(arr[0].x + arr[0].y + arr[1].x + arr[1].y == 50);
}
