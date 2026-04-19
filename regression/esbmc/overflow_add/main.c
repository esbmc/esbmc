#include <limits.h>

int main()
{
  int a = INT_MAX, b = 1;
  int add_overflow = a + b; // Undefined behavior
  return 0;
}
