// An exception leaving a function destroys that function's automatic objects.
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

bool destroyed = false;
void g()
{
  Guard x(&destroyed);
  throw E();
}

int main()
{
  try
  {
    g();
  }
  catch (E &)
  {
    __ESBMC_assert(destroyed, "callee local destroyed as exception leaves g");
  }
  return 0;
}
