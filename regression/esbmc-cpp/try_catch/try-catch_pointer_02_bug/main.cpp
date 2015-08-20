#include<cassert>

int main()
{
  try {
    int x = 5;
    int *py = &x;

    throw py;
  }
  catch(void*) { assert(0); }
  return 0;
}
