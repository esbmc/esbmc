// esbmc/esbmc#4377 (C++17): the `switch` init-statement form, the twin of the
// `if` case in github_4377_if_init_statement.
#include <cassert>

int main()
{
  switch (int x = 7; 1)
  {
  case 1:
    assert(x == 7);
    break;
  default:
    assert(0);
  }

  int hit = 0;
  switch (int sel = 2; sel)
  {
  case 1:
    hit = 10;
    break;
  case 2:
    hit = sel * 100;
    break;
  }
  assert(hit == 200);
  return 0;
}
