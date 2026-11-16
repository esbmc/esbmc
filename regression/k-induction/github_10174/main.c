/* github #10174: the block after the outer loop's back edge starts with
 * another entry jump's havocs, which must not capture the loop exit edge. */
#include <assert.h>
int main()
{
  int d = 1, x, t;
top:
  if (d <= 3)
  {
    if (d <= 1)
    {
      x = 1;
    two:
      if (x >= 1)
      {
        t = 1;
      l:
        assert(t != 0);
        d = d + 1;
        goto top;
      b:
        t = 0;
        goto l;
      }
      else
      {
        goto b;
      }
    }
    else
    {
      x = 2;
      goto two;
    }
  }
  return 0;
}
