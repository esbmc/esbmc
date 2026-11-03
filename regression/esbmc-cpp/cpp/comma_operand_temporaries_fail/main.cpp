#include <cassert>

int dtors = 0;

struct T
{
  ~T()
  {
    dtors++;
  }
};

int seen()
{
  return dtors;
}

int main()
{
  int a = (T(), seen());
  assert(a == 1);
  return 0;
}
