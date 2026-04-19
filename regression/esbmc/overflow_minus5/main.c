#include <limits.h>

int main()
{
  int c = INT_MIN, d = 1;
  int sub_underflow = c - d; // Undefined behavior
  return 0;
}
