// A dynamic exception specification throw(char): the function throws an int,
// which is NOT permitted, so the exception escapes the function out-of-spec.
// The lowering asserts at the epilogue that any in-flight exception's type is
// in the spec; an int is not, so the specification is violated ([except.spec]).
// (This mirrors the imperative path, which reports an out-of-spec escape as
// "not allowed by declaration"; unexpected() dispatch is modelled on neither.)
void f() throw(char)
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
    return 1;
  }
  return 0;
}
