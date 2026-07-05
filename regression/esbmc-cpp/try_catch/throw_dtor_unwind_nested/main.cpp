// Nested try: a throw in the inner try not caught there propagates to the outer
// handler, destroying the inner-try local but not the outer-try local that is
// declared after the inner try would have completed.
struct Guard
{
  bool *f;
  Guard(bool *p) : f(p)
  {
  }
  ~Guard()
  {
    *f = true;
  }
};
struct A
{
};
struct B
{
};

int main()
{
  bool inner_d = false;
  try
  {
    try
    {
      Guard inner(&inner_d);
      throw B(); // not caught by inner catch(A&)
    }
    catch (A &)
    {
    }
  }
  catch (B &)
  {
    __ESBMC_assert(inner_d, "inner-try local destroyed during propagation");
  }
  return 0;
}
