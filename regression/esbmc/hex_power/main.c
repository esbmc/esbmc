#include <stdio.h>

/*http://www.exploringbinary.com/hexadecimal-floating-point-constants/
* The suffix ‘p-4’, which represents the power of two, written in decimal: 2^-4
*/
int main()
{
  static volatile const double Tiny = 0x1p-1022;
  printf("%.20f\n", Tiny);
  return 0;
}
