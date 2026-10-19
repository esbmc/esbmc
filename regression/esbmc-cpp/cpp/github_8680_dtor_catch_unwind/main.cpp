#include <cassert>
struct T
{
  ~T() noexcept(false)
  {
    try
    {
      throw 2;
    }
    catch (int)
    {
    }
  }
};
int main()
{
  int caught = 0;
  try
  {
    T t;
    throw 1;
  }
  catch (int k)
  {
    caught = k;
  }
  assert(caught == 1);
  return 0;
}
