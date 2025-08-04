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

      __ESBMC_loop_invariant(sn == (i - 1) * a);
      for(i=1; i<=n; i++)
        sn = sn + a;

      res = sn;
    }
};

int N=nondet_int();

int main() 
{
  Sum s(N);
  assert(s.res==N*a || s.res == 0);

  return 0;
}
