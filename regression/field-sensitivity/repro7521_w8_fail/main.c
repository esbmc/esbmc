/* #7521 field-sensitivity corpus: the #7521 reproducer, a field written once and read in a loop.
 * The struct has 8 fields; F<n>(p) declares n padding fields. */
#include <assert.h>

struct temp
{
  int x, a1, a2, a3, a4, a5, a6, a7;
};

int fun()
{
  struct temp temp1;
  temp1.x = 1;
  for (int i = 0; i < 20; i++)
    for (int j = 0; j < 10; j++)
      if (temp1.x)
        assert(j < 9);
  return 0;
}

int main()
{
  for (int k = 0; k < 100; k++)
    fun();
  return 0;
}
