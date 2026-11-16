#include <assert.h>

int main()
{
  int r = 0;
  for (int pass = 0; pass < 3; ++pass)
  {
    switch (pass)
    {
    case 0:
    {
      int i = 5;
      r += i;
      continue;
    case 1:
      i = 7;
      switch (i)
      {
      case 7:
      {
        int j = 1;
        r += j;
        break;
      default:
        j = 100;
        r += j;
      }
      }
      r += i;
      continue;
    }
    default:
    {
      int k;
      k = 2;
      r += k;
    }
    }
  }
  assert(r == 15);
}
