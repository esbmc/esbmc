// A rethrow inside a handler unwinds to the OUTER try level, destroying the
// handler-scope locals constructed before the rethrow.
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

bool handler_local_destroyed = false;

void inner()
{
  try
  {
    throw E();
  }
  catch (E &)
  {
    Guard h(&handler_local_destroyed);
    throw; // rethrow: must unwind h
  }
}

int main()
{
  try
  {
    inner();
  }
  catch (E &)
  {
    __ESBMC_assert(handler_local_destroyed, "handler local destroyed on rethrow");
  }
  return 0;
}
