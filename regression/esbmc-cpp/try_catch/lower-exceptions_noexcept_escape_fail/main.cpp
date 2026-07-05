// An exception thrown directly in a noexcept function escapes the no-throw
// boundary -> std::terminate ([except.spec]). The lowering asserts that no
// exception is in flight at the function epilogue.
struct E
{
  int v;
  E(int a) : v(a)
  {
  }
};

void f() noexcept
{
  throw E(5);
}

int main()
{
  try
  {
    f();
  }
  catch (...)
  {
    return 1;
  }
  return 0;
}
