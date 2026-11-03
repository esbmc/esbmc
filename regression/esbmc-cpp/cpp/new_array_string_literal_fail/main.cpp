#include <cassert>

int main()
{
  char *p = new char[4]{"ab"};
  assert(p[1] == 'b');
  assert(p[2] == 'c');
  delete[] p;
}
