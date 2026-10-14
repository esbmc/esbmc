#include <assert.h>
enum Color { RED, GREEN, BLUE };
int main()
{
  enum Color c = BLUE;
  assert(c == GREEN); // BLUE (2) != GREEN (1)
  return 0;
}
