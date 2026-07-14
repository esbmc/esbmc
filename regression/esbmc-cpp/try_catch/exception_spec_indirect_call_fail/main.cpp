// For an indirect (function-pointer) call, the specification enforced is that
// of the resolved callee. Calling f through a pointer still terminates when its
// noexcept boundary is crossed by an exception.
void f() noexcept
{
  throw 5;
}

typedef void (*fp)();

int main()
{
  fp p = f;
  try
  {
    p();
  }
  catch (...)
  {
  }
  return 0;
}
