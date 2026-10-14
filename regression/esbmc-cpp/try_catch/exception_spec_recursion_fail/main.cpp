// The specification is enforced per call frame. A recursive noexcept function
// that throws at the base of the recursion violates its boundary regardless of
// how deep the throwing frame is; the catch(...) in main is never reached.
void f(int n) noexcept
{
  if (n == 0)
    throw 5;
  f(n - 1);
}

int main()
{
  try
  {
    f(3);
  }
  catch (...)
  {
  }
  return 0;
}
