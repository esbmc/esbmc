#include <cassert>

unsigned nondet_uint();

int main()
{
  char *p = new char[4]{"ab"};
  assert(p[0] == 'a' && p[1] == 'b' && p[2] == 0 && p[3] == 0);
  delete[] p;

  wchar_t *w = new wchar_t[5]{L"hi"};
  assert(w[0] == L'h' && w[1] == L'i' && w[4] == 0);
  delete[] w;

  unsigned n = nondet_uint();
  __ESBMC_assume(n >= 3 && n <= 6);
  char *r = new char[n]{"cd"};
  assert(r[0] == 'c' && r[1] == 'd' && r[n - 1] == 0);
  delete[] r;
}
