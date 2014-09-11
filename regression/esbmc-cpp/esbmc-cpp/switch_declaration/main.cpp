#include <cassert>

int main()
{
  switch(int x=0)
  {
    case true:
      assert(0);
      break;

    case false:
      break;

    default:
      assert(0);
      break;
  }

  return 0;
}
