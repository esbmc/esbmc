// Loop body has a __VERIFIER_assert call. Termination depends only on
// the counter `k` decrementing the guard `k < 0x0fffffff`; the assert
// is irrelevant (either it holds and the path continues, or it fails
// and the program aborts — both terminate). The certifier must skip
// the assert call as if it were a no-op and prove termination via the
// scalar measure `0x0fffffff - k`. Lifted from loop-crafted/simple_vardep_1.

extern int __VERIFIER_nondet_int(void);
extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *)
    __attribute__((__nothrow__, __leaf__)) __attribute__((__noreturn__));
void reach_error()
{
    __assert_fail("0", "main.c", 3, "reach_error");
}
void __VERIFIER_assert(int cond)
{
    if (!cond) {
    ERROR: { reach_error(); abort(); }
    }
    return;
}

int main(void)
{
    unsigned int i = 0;
    unsigned int j = 0;
    unsigned int k = 0;
    while (k < 0x0fffffff) {
        i = i + 1;
        j = j + 2;
        k = k + 3;
        __VERIFIER_assert(k == (i + j));
    }
    return 0;
}
