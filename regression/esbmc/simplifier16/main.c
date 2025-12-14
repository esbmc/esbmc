#include <assert.h>

// Multiplication by constant on both sides
// (x * c) == (y * c) -> x == y when c != 0
int main() 
{
  int x, y;
    
  // Should simplify to x == y
  assert(((x * 5) == (y * 5)) == (x == y));
  assert(((3 * x) == (3 * y)) == (x == y));

  return 0;
}
