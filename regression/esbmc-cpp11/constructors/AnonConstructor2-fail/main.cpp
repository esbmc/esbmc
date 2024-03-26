#include <cassert>

class Other
{
public:
  int x;

  Other()
  {
    x = 2;
  }
};

int main()
{
  // Define an unnamed struct
  struct
  {
    int field1;
    float field2;
    Other field3;
  } unnamed_struct;

  // Access and verify the fields of the unnamed struct
  assert(unnamed_struct.field3.x == 3);
  return 0;
}