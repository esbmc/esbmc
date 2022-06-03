#include <cassert>
int counter = 1;

class T
{
public:
  int z;

  T();
};

T::T()
{
  z = counter;
  counter++;
}

T a, b;

int main()
{
  assert(counter == 3);
  assert(a.z == 1);
  assert(b.z == 2);
}
