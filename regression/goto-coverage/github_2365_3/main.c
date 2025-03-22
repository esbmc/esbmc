#include <assert.h>
#include <stdbool.h>

int a;
int b;

struct foo
{
  _Bool x;
};

int main()
{
  int a = nondet_int();
  struct foo *npe = 0;
  __ESBMC_assume(a != 1);
  if (a == 1)
  {
    if (npe->x)
    {
      assert(0 && "if");
    }
    else
    {
      assert(1 && "else");
    }
  }
  else
    assert(1 && "else");
}