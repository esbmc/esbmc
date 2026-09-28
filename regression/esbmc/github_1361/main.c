/* github.com/esbmc/esbmc/issues/1361: both assertions must reach the report,
   in one table, with an id each. */
#include <assert.h>

int main()
{
  for (int i = 0; i < 2; i++)
  {
    assert(2 == 3);
  }
  assert(1 == 2);
  return 0;
}
