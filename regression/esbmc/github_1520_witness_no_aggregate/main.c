/* Regression for #1520 (and #1471): an aggregate value (array / nested array /
 * struct, or any ESBMC-internal placeholder such as nondet_symbol / ARRAY_OF)
 * must never appear inside a GraphML witness assumption, because the SV-COMP
 * witness validators (CPAchecker, cpa-witness2test) cannot parse a brace
 * initialiser and reject the whole automaton.
 *
 * Here `status` is a zero-initialised global nested array. The violating step
 * reads a single scalar element, so the witness should keep the scalar
 * assumption (x == 0) and drop the unparseable `status == { { 0, ... } }`. */
unsigned char status[2][3];

int main()
{
  unsigned char x = status[1][2];
  assert(x);
}
