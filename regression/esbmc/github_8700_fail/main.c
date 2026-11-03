/* github #8700: int and _BitInt are not promoted, so ++ on them can overflow */
#include <limits.h>
int main(void)
{
  signed char c = 127;
  c++;
  int i = INT_MAX;
  i++;
  _BitInt(7) b = 63;
  ++b;
  return 0;
}
