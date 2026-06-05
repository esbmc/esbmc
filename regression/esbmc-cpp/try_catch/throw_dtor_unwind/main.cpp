// C++ stack unwinding: an automatic object in a try block is destroyed when an
// exception is thrown out of it ([except.ctor]).
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
    __ESBMC_assert(destroyed, "g destroyed during unwinding");
  }
  return 0;
}
