// #2320: #pragma comment yields a PragmaCommentDecl that has no runtime form.
#include <cassert>
#pragma comment(lib, "m")

int main()
{
  int x = 1;
  assert(x == 2);
  return 0;
}
