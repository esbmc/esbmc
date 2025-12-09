// TC description:
//  - This TC is just a variant of template_1, using float instead of int
//    for the type parameter of class template
//  - lvalue reference to an array in function template argument
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

int main()
{
    FixedArray25<float, char> sMyArray;

    Value(sMyArray, 10) = 5.0;
    assert(sMyArray.anValue[10] == 5.0);
    Value(sMyArray, 15) = 10.0;
    assert(sMyArray.anValue[15] == 10.0);
    assert(sMyArray.anValue[10] == 5.0);

    return 0;
}
