// #2320: #pragma comment yields a PragmaCommentDecl that has no runtime form.
#pragma comment(lib, "m")

int main()
{
  int x = 1;
  __ESBMC_assert(x == 2, "x is two");
  return 0;
}
