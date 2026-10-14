// Body __VERIFIER_assume is load-bearing: without it the certifier
// cannot bound the measure below.
//
// Loop: `while (n < MAX) { __VERIFIER_assume(n >= MAX - 1000); n++; }`.
// Guard `n < MAX` → measure m = MAX - n, L = 1 (strict).
// Without the body assume, n could be a deeply negative int and the
// measure MAX - n would be arbitrarily large; the bounded-below
// obligation `(n < MAX) ∧ (MAX - n < 1)` (= `n > MAX - 1`) is UNSAT
// since `n < MAX` ⇒ `n ≤ MAX - 1`, so actually bounded-below
// discharges by integer reasoning alone. The assume is redundant for
// THIS shape — keep test minimal but exercise the ASSUME-recording
// path. K-induction can't unwind MAX iterations within k-step 2, so
// without the ranking discharge it falls back to UNKNOWN. With body
// ASSUME forwarded to the certifier the ranking path catches it.

extern int __VERIFIER_nondet_int(void);
extern void __VERIFIER_assume(int);

int main(void)
{
    int n = __VERIFIER_nondet_int();
    while (n < 2000000000) {
        __VERIFIER_assume(n >= 1000000000);
        n = n + 1;
    }
    return 0;
}
