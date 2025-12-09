#include <limits.h>

int main()
{
  int g = INT_MIN, h = -1;
  int div_overflow = g / h; // Undefined behavior
  return 0;
}
