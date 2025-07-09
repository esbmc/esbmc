#include <assert.h>
#include <float.h>

int main() 
{
  double x = DBL_MIN;
  double y = -DBL_MIN;
  double result = x + y;

  assert(result > 0.0); // Fails: exact zero due to cancellation
  return 0;
}
