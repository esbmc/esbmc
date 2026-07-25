/* replace_struct_ptr_havoc_fail (issue #6356):
 * Negative counterpart of replace_struct_ptr_havoc_pass. The ensures only
 * bounds the field (>= 0), so the pointer-parameter havoc must leave it
 * nondet; asserting it kept its old value must therefore FAIL. Guards against
 * the fix regressing into a havoc that doesn't actually nondet the struct
 * (which would make the ensures ASSUME vacuous and spuriously verify).
 * Expected: VERIFICATION FAILED
 */
typedef struct { int received; } S;

void f(S *self)
{
    __ESBMC_requires(self != 0);
    __ESBMC_ensures(self->received >= 0);
    self->received = 5;
}

int main(void)
{
    S s;
    s.received = 42;
    f(&s);
    __ESBMC_assert(s.received == 42, "must fail: havoc makes received nondet");
    return 0;
}
