#include<cassert>

template< class T, class T1>
class FixedArray25
{
  public:
    T anValue[25];
};

// Returns a reference to the nIndex element of rArray
template< class T, class T1>
T& Value( FixedArray25<T, T1> &rArray, int nIndex)
{
  return rArray.anValue[nIndex];
}

template< class T>
T& Value1(T &value)
{
  return value;
}

int main()
{
    FixedArray25<float, char> sMyArray;
    FixedArray25<int, char> sMyArray1;

    int x=5;
    assert(Value1(x)==5);

    Value(sMyArray, 10) = 5.0;
    assert(sMyArray.anValue[10] == 5.0);
    Value(sMyArray, 15) = 10.0;
    assert(sMyArray.anValue[15] == 10.0);
    assert(sMyArray.anValue[10] != 5.0);

    return 0;
}
