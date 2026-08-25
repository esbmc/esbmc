/* A declaration's initialiser converts to the declared type
   (clang_c_adjust::adjust_decl's trailing gen_typecast). The _ExtInt operands
   promote to int for the addition, so without the conversion back the
   initialiser assigns an int to a 10-bit target and the solver is handed
   mismatching sorts -- bitvector_04 aborted inside bitwuzla. Nondet operands:
   a constant initialiser folds before the mismatch can reach the encoder. */
#include <assert.h>

int nondet_int(void);

int main(void)
{
  _ExtInt(10) x = nondet_int();
  _ExtInt(10) y = nondet_int();
  _ExtInt(10) z = x + y;
  /* The declared width bounds the initialised value; unconverted, the
     initialiser is an int and the encoder never gets this far. */
  assert(z >= -512 && z <= 511);
  return 0;
}
