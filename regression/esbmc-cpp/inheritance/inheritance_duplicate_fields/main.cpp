#include <cassert>
#include <cstring>

struct Ai
{
  int i;
};

struct Bii : Ai
{
  int i;
};

int main()
{
  Bii bii;
  bii.i = 1;
  bii.Ai::i = 2;
  assert(bii.i == 1);
  assert(bii.Ai::i == 2);

  return 0;
}