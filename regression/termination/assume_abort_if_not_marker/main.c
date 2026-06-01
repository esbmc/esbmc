/* assume_abort_if_not(cond) marker pattern, lifted from the
 * seq-mthreaded SV-COMP benchmarks. The wrapper is defined as
 * `if (!cond) abort();` — the same shape as __VERIFIER_assert but
 * a different name. Aborts iff cond is false.
 *
 * insert_abort_call_markers_for_function recognises this wrapper
 * (alongside __VERIFIER_assert and a benchmark-overridden `assert`)
 * and emits ASSERT(cond) before each call. With the marker in
 * place, FC discharges the loop because the abort short-circuits
 * every iteration past the bound. Expected verdict:
 * VERIFICATION SUCCESSFUL. */

extern void abort(void);

void assume_abort_if_not(int cond)
{
  if (!cond)
    abort();
}

int main()
{
  int x = 0;
  while (1)
  {
    x++;
    assume_abort_if_not(x < 5);
  }
}
