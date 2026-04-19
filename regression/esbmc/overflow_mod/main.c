#include <limits.h>

int main()
{
  int i = INT_MIN, j = -1;
  int mod_overflow = i % j; // Undefined behavior
  return 0;
}
