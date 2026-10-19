#include <cassert>
unsigned nondet_uint();
enum class K : unsigned char
{
  X = 1,
  Y = 200
};
int main()
{
  K k = static_cast<K>(nondet_uint());
  switch (k)
  {
  case K::Y:
    assert(false);
    break;
  default:
    break;
  }
  return 0;
}
