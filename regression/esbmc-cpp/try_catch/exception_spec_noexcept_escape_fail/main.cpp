// An exception that escapes a noexcept function crosses the function boundary
// and must invoke std::terminate. The surrounding catch(...) in main is never
// reached, so this verification fails.
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
