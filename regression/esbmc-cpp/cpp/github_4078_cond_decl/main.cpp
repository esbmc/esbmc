// A condition that declares a variable of class type is tested through the
// variable's conversion to bool. The frontend used to take the declaration
// itself as the condition, which discarded that conversion and handed the
// solver a struct where a boolean was required (issue #4078).
struct c
{
  int a;
  c(int x) : a(x)
  {
  }
  operator bool()
  {
    return a != 0;
  }
};

int k = 3;

int main()
{
  int hit = 0;
  if (c b = 0)
    hit = 1;
  __ESBMC_assert(hit == 0, "a zero-valued condition variable is false");

  if (c b = 1)
    hit = 2;
  __ESBMC_assert(hit == 2, "a non-zero condition variable is true");

  // The variable is rebuilt each iteration, so the loop sees k decrease.
  int n = 0;
  while (c b = k)
  {
    n++;
    k--;
  }
  __ESBMC_assert(n == 3, "while re-evaluates the declaration each iteration");

  int m = 0;
  for (int j = 3; c b = j; j--)
    m++;
  __ESBMC_assert(m == 3, "for re-evaluates the declaration each iteration");
  return 0;
}
