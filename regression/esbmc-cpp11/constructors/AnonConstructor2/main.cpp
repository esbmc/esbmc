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
  } unnamed_struct = {10, 20.5f};

  struct
  {
    int field1;
    float field2;
    Other field3;
  } unnamed_struct2;

  // Access and verify the fields of the unnamed struct
  assert(unnamed_struct.field1 == 10);
  assert(unnamed_struct.field2 == 20.5f);
  assert(unnamed_struct.field3.x == 2);

  // Access and verify the fields of the unnamed struct
  unnamed_struct2.field1 = 99;
  unnamed_struct2.field2 = 330.0f;
  unnamed_struct2.field3.x = 55;
  assert(unnamed_struct2.field1 == 99);
  assert(unnamed_struct2.field2 == 330.0f);
  assert(unnamed_struct2.field3.x == 55);

  return 0;
}