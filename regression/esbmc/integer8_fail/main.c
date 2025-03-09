#include <limits.h>

int main() 
{
  int a = INT_MIN;
  // This operation results in signed integer overflow
  int result = -a;
  return 0;
}

