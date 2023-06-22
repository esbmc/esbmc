/*
 * TC description:
 *  class template, UDT(class) for type parameter, implicit template instantiation, data member has dependent type
 */
#include<cassert>

// define Z<>
template <typename T>
class Z
{
public:
  typename T::f some;
};

// Forward declaration of FF
class FF;

// make an instance of Z<FF>
typedef Z<FF> my_Z;

// Declare FF
class FF
{
public:
  typedef int f;
};

// trigger elaboration of Z<FF>
my_Z z;

int main()
{
  z.some = 39;
  assert(z.some==39);
  return 0;
}
