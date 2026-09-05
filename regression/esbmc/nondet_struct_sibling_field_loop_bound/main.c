/* A FOR loop with a fully static bound (1 TO 5) whose induction variable is
 * a sibling field of a nondet-assigned struct member. Before the fix, any
 * nondet write into one field made the whole struct's SSA value opaque, so
 * the loop guard on the (otherwise perfectly concrete) counter field never
 * folded to a literal: without an explicit --unwind, ESBMC's automatic
 * loop-bound search never recognises the loop as bounded and unwinds it
 * forever. This is a correctness sibling of #7524, which fixed a
 * performance-only consequence of the same whole-struct-not-per-field SSA
 * caching. */
typedef struct
{
  float REAL_1;
  int INT_1;
} VAR_t;
VAR_t VAR;
float nondet_val;

int main(void)
{
  VAR.INT_1 = 0;
  VAR.REAL_1 = 0.0;
  nondet_val = nondet_float();
  for (VAR.INT_1 = 1; VAR.INT_1 <= 5; VAR.INT_1 = VAR.INT_1 + 1)
    VAR.REAL_1 = nondet_val;
  return 0;
}
