// An exception thrown by a callee propagates into the noexcept caller and
// escapes its no-throw boundary -> std::terminate.
struct E
{
  int v;
  E(int a) : v(a)
  {
  }
};

void g()
{
  throw E(1);
}

void f() noexcept
{
  g();
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
