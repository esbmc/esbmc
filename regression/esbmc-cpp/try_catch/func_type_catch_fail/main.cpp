// Negative variant: the int is caught by `catch (int)`, not the function-type
// clause, so caught == 42 and asserting it equals -1 must fail.
#include <exception>
#include <cassert>
using std::exception;

int main()
{
  int caught = 0;
  try
  {
    throw 42;
  }
  catch (exception())
  {
    caught = -1;
  }
  catch (int e)
  {
    caught = e;
  }
  assert(caught == -1);
  return 0;
}
