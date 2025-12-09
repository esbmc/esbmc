#include <cassert>

int main()
{
  // Define and initialize an unnamed struct
  struct
  {
    int field1;
    float field2;
  } unnamed_struct = {10, 20.5f};

  // Access and verify the fields of the unnamed struct
  assert(unnamed_struct.field1 == 10);
  assert(unnamed_struct.field2 == 20.5f);

  return 0;
}