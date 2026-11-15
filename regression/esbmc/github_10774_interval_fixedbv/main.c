// --interval-analysis simplifies the unused libm models <math.h> pulls in,
// whose remainder() call must not become a floating-point-only node.
#include <math.h>
int main()
{
  return 0;
}
