#include <assert.h>

int main()
{
  int r = 0;
  for (int pass = 0; pass < 2; ++pass)
  {
    if (pass == 1)
      goto WAIT;
    {
      int i = 5;
      r = i;
      continue;
    WAIT:
      i = 7;
      r += i;
    }
  }
  assert(r == 12);

  {
    int a = 1;
    goto B;
  }
  {
    int b;
  B:
    b = 2;
    assert(b == 2);
  }
}
