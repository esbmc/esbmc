#include <assert.h>

struct
{
  _ExtInt(17) field_1;
  int field_2;
} s;

int main()
{
  // Check that the field is at the correct alignment and can be dereferenced
  assert(*(&s.field_2) == *(int *)((void *)&s.field_1 + 4));
}
