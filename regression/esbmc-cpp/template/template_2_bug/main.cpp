// TC description:
//  lvalue reference argument to an integral in function template
#include<cassert>

template< class T>
T& Value1(T &value)
{
  return value;
}

int main()
{
    int x=5;
    assert(Value1(x)!=5); // should be 5

    return 0;
}
