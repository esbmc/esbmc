#include<cassert>

int main()
{
  try {
    int array[5];
    throw array;
  }
  catch(int*) { assert(0); }
  return 0;
}
