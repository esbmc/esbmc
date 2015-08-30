#include <cassert>

int main()
{
  switch(int x=5)
  {
    case 0:
      assert(0);
      break;

    case 1:
      assert(0);
      break;

    default:
      break;
  }

  return 0;
}
