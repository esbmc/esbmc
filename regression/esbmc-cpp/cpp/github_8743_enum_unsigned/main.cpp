#include <cassert>
unsigned nondet_uint();
enum K : unsigned short
{
  Y = 40000
};
int main()
{
  K k = static_cast<K>(nondet_uint());
  if (k == Y)
    assert(static_cast<unsigned short>(k) == 40000);
  else
    assert(static_cast<unsigned short>(k) != 40000);
  return 0;
}
