/* Negative twin of nondet_struct_sibling_field_loop_bound: same sibling-field
 * loop, but with an assertion on the counter's final value that must be
 * refuted (the loop exits with INT_1 == 6, not 5). This proves the fix's
 * folding is genuinely tracking the concrete field's value -- not merely
 * suppressing the hang by giving up and reporting success -- and that ESBMC
 * still terminates and finds the violation, rather than unwinding forever,
 * once the sibling field can fold past the nondet one. */
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
  __ESBMC_assert(VAR.INT_1 == 5, "loop counter wrongly asserted 5 after natural exit");
  return 0;
}
