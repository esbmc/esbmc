// A dynamic exception specification throw(int) is violated by throwing a char.
// Crossing f's boundary with a disallowed type must invoke std::unexpected,
// which by default terminates: the verification fails.
void f() throw(int)
{
  throw 'c';
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
