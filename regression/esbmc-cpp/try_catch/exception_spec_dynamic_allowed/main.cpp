#include <cassert>

// A dynamic exception specification throw(int) is satisfied when an int is
// thrown. The exception crosses f's boundary without violating the spec and is
// delivered normally to the handler in main.
void f() throw(int)
{
  throw 5;
}

int main()
{
  int caught = 0;
  try
  {
    f();
  }
  catch (int)
  {
    caught = 1;
  }
  assert(caught == 1);
  return 0;
}
