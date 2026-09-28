// The bundled <stdio.h> omitted these ISO C declarations. In C, clang's builtin
// table masked it; in C++ there is no such fallback, so the program was a
// PARSING ERROR (#7548).
#include <cassert>
#include <cstdio>

int main()
{
  perror("x");

  char name[L_tmpnam];
  assert(sizeof(name) == L_tmpnam);

  FILE *f = tmpfile();
  if (f != nullptr)
  {
    setbuf(f, nullptr);
    (void)setvbuf(f, nullptr, _IONBF, 0);
    (void)ungetc('a', f);

    fpos_t pos;
    if (fgetpos(f, &pos) == 0)
      (void)fsetpos(f, &pos);
  }
  return 0;
}
