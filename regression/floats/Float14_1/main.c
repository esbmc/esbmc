#include <math.h>

int main()
{
  float temp;
  
  temp = 1.8e307f + 1.5e50f;	// should produce overflow -> +infinity (according to standard)
  assert(isinf(temp));
  
  float x;
  
  x=temp-temp;
  
  // should be +inf
  assert(isinf(temp));
}
