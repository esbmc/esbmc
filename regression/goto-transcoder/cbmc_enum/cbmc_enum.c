#include <assert.h>
enum Color { RED, GREEN, BLUE };
enum Flags { F_A = 1, F_B = 2, F_C = 4 };
struct Pixel { enum Color c; int v; };
int main()
{
  enum Color c = GREEN;
  assert(c == 1);
  assert(BLUE - RED == 2);
  assert((F_A | F_C) == 5);
  struct Pixel p = { BLUE, 10 };
  assert(p.c == 2);
  return 0;
}
