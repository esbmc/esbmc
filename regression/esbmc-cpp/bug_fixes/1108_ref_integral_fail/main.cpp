#include<cassert>

int& Value1(int &value)
{
  return value;
}

int main()
{
    int x=5;
    assert(Value1(x)==10); // should be 5
    return 0;
}
