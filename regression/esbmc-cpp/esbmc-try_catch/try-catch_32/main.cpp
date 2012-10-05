#include <cassert>

int main()
{
  try {
    throw 5;
    assert(0);
  }
  catch(int e) {
    return 1;
  }
  return 0;
}
