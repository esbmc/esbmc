#include <ctype.h>
#include <limits.h>
 
int main(void)
{
  int hex = 0;
  for (int ndx=0; ndx<=UCHAR_MAX; ndx++)
    if (isxdigit(ndx)) hex++;

  assert(hex == 20);
}
