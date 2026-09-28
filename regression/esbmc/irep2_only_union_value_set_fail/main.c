/* The union assignment still carries its value: reading back the member that
   was written must not be excused. */
#include <assert.h>

union bits
{
  int : 5;
};

union payload
{
  int i;
  char c;
};

int main(void)
{
  union bits b = {};
  union payload p = {7};
  union payload q = p;
  assert(q.i == 8);
  (void)b;
  return 0;
}
