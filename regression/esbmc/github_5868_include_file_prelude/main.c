/* github #5868 gap 5: the intrinsic prelude was appended to clang's predefines
   buffer after the `#include` lines --include-file generates. esbmc_action.h is
   shared by the C and C++ frontends, so the C path needs the same guard. */
int main()
{
  assert(helper() == 3);
  return 0;
}
