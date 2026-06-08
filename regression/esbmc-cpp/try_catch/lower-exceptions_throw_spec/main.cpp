// A dynamic exception specification throw(char): the function throws a char,
// which IS permitted, so it escapes legally and is caught by main. The lowering
// enforces the spec at the function epilogue (typeid in { id(char) }), so an
// in-spec escape passes the check and propagates normally.
void f() throw(char)
{
  throw 'x';
}

int main()
{
  try
  {
    f();
  }
  catch (char)
  {
    return 1;
  }
  return 0;
}
