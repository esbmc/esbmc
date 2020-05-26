#include <stdio.h>
#include <ctype.h>
#include <limits.h>
 
int main(void)
{
  int sp = 0;
  for (int ndx=0; ndx<=UCHAR_MAX; ndx++)
    if (isspace(ndx)) sp++;

  assert(sp == 6);
}
