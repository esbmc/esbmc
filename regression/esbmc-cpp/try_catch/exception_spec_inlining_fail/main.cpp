// Even under full inlining, a function with a restrictive exception
// specification keeps its own frame so the boundary can be enforced. Were f
// inlined into main, the noexcept boundary would be lost and the escape would
// go undetected. Run with --full-inlining to exercise that path.
void f() noexcept
{
  throw 5;
}

int main()
{
  try
  {
    f();
  }
  catch (...)
  {
  }
  return 0;
}
