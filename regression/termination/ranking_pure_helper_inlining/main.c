// Loop body has two FUNCTION_CALLs to a pure helper `foo` that
// decrements a global scalar `x`. Without inlining, the certifier
// rejects the loop because FUNCTION_CALL terminates the recognized
// straight-line span. With targeted pure-helper inlining, `foo()` is
// expanded into its body (`x = x - 1;`) at each call site, the
// span becomes `x = x - 1` regardless of the inner branch, and the
// measure `x` strictly decreases.
//
// Lifted from loops/trex02-1.c.

extern int __VERIFIER_nondet_int(void);

int x;

void foo(void)
{
    x--;
}

int main(void)
{
    x = __VERIFIER_nondet_int();
    while (x > 0) {
        int c = __VERIFIER_nondet_int();
        if (c)
            foo();
        else
            foo();
    }
    return 0;
}
