#include<assert.h>

int main(void)
{
  bool b = false;
  reinterpret_cast<char&>(b) = 'a';
  assert(b);
  return 0;
}
