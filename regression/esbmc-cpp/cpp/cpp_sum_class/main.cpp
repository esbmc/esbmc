#include <cassert>

#define a (2)
int nondet_int();

class Sum
{
  public:
    int res;

    Sum(int N)
    {
      int i, n=N, sn=0;

      for(i=1; i<=n; i++)
        sn = sn + a;

      res = sn;
    }
};

int N=nondet_int();

int main()
{
  __ESBMC_assume(N >= 0 && N <= 10);
  Sum s(N);
  assert(s.res==N*a || s.res == 0);

  return 0;
}
