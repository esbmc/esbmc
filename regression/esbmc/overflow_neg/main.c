#include <limits.h>

int main()
{
  int k = INT_MIN;
  int neg_overflow = -k; // Undefined behavior
  return 0;
}
