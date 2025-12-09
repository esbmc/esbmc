#include <assert.h>

int main()
{
  int other = 0;
  struct nested
  {
    int &y;
  };
  nested n{other};
  assert(++(n.y) == 1111);
}
