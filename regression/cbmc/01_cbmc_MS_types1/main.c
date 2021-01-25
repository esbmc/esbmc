#include <stdint.h>

int main()
{
  int8_t i1;
  int16_t i2;
  int32_t i3;
  int64_t i4;
  
  assert(sizeof(i1)==1);
  assert(sizeof(i2)==2);
  assert(sizeof(i3)==4);
  assert(sizeof(i4)==8);
}
