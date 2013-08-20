#include <cassert>

int main()
{
  switch(int x=0)
  {
    case 0:
      break;

    case 1:
      assert(0);
      break;

    default:
      assert(0);
      break;
  }

  return 0;
}
