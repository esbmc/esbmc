#include<cassert>

int main()
{
  try {
    int array[];
    throw array;
  }
  catch(int[]) { assert(0); }
  return 0;
}
