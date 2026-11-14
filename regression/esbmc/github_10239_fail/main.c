/* argv[3] and argv[4] are separate strings that may differ, and reading past
   the end of one is still caught (esbmc/esbmc#10239). */
#include <assert.h>
int main(int argc, char **argv)
{
  if (argc == 5)
  {
    assert(argv[3][0] == argv[4][0]);
    return argv[4][256];
  }
  return 0;
}
