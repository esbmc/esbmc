#include<cassert>

int main()
{
  try {
    int array[];
    throw array;
  }
  catch(int[5]) { assert(0); }
  return 0;
}
