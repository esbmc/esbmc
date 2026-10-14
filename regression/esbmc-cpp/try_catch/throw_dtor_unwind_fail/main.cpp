// Negative: asserting the object was NOT destroyed must fail, since unwinding
// does destroy it.
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
struct E
{
};

int main()
{
  bool destroyed = false;
  try
  {
    Guard g(&destroyed);
    throw E();
  }
  catch (E &)
  {
    __ESBMC_assert(!destroyed, "expected to fail: object IS destroyed");
  }
  return 0;
}
