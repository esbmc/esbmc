#ifdef PRINT
#include <stdio.h>
#endif

int rec(int x)
{
  if( x == 0 )
    return x;
  return rec(x-1) +1;
}

int main()
{
  int res = 0;

  res = rec(3);
  res = res + rec(3);
  res = res + rec(3);
  res = res + rec(3);

#ifdef PRINT
  printf("res = %d\n",res);
#else
  assert( res == 12 );
#endif

  return res;
}
