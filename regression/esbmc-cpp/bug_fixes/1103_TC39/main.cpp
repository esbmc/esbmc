// This is the original TC for cbmc/Templates, WITH zero init assumption
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
  assert(z.some==0);
  return 0;
}
