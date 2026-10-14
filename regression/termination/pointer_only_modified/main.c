/* Pointer-only-modified loop (strrchr-style). The loop's only modified
 * variable is the pointer `t`. Pre-merge HEAD had a dedicated gate
 * (has_pointer_only_loop) disabling the inductive step for this shape;
 * the upstream k-induction deep-dive merge removed it on the claim
 * that the value-set assume made it unnecessary. The assume protects
 * pointer aliasing (post-havoc `t` stays inside its pre-havoc points-
 * to set) but does NOT constrain the pointed-to data — for loops
 * whose termination depends on what's read through the pointer (here,
 * the `*t == 0` exit condition), IS sees a state where `*t != 0` for
 * every k it unwinds and reports non-termination spuriously.
 *
 * goto_terminationt's loop_is_is_unreliable predicate now sets
 * disable-inductive-step for the whole program when it sees a loop
 * whose modified set is non-empty but consists entirely of pointer-
 * typed entries. The expected verdict here is therefore UNKNOWN
 * (FC can't unwind a nondet-length string, IS is disabled), not
 * the wrong VERIFICATION FAILED IS used to produce. */

extern int __VERIFIER_nondet_int(void);

int main()
{
  int in_len = __VERIFIER_nondet_int();
  if (in_len < 1)
    return 1;
  char *in = (char *)__builtin_alloca(in_len);
  in[in_len - 1] = 0;

  /* Modifies only the pointer `t`; `*t` is read, never written. */
  char *t = in;
  while (*t != 0)
    ++t;

  return 0;
}
