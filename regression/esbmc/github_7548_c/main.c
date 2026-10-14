// fpos_t and L_tmpnam are a type and an object-like macro, so clang's builtin
// table cannot mask their absence: these failed in C too (#7548).
#include <assert.h>
#include <stdio.h>

int main(void)
{
  char name[L_tmpnam];
  assert(sizeof(name) == L_tmpnam);

  FILE *f = tmpfile();
  if (f != 0)
  {
    fpos_t pos;
    if (fgetpos(f, &pos) == 0)
      (void)fsetpos(f, &pos);
  }
  return 0;
}
