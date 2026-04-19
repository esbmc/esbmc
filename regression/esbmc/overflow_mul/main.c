#include <limits.h>

int main()
{
  int e = INT_MAX, f = 3;
  int mul_overflow = e * f; // Undefined behavior
  return 0;
}
