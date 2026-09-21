/* mainc matches the prefix declare_argc_argv's call sites test, so without an
   entry-point check its `char` first parameter becomes argc''s type and
   clang_c_main's arithmetic on argc' is built at the wrong width
   (esbmc/esbmc#4715). */
#include <assert.h>

int mainc(char a, char **b);

int main(int argc, char **argv)
{
  assert(argc <= 127);
  return 0;
}
