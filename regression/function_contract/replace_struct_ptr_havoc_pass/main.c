/* replace_struct_ptr_havoc_pass (issue #6356):
 * --replace-call-with-contract on a struct-pointer parameter whose contract
 * has BOTH a null-check `requires` and a member-deref `ensures` used to abort
 * with "Unexpected type ID symbol reached SMT conversion". The pointer-param
 * havoc built `*arg = nondet(pointee)` without resolving the struct "tag"
 * (symbol_type2t) pointee, so a symbol type reached SMT sort conversion.
 * Here the ensures pins the field, so the post-state is known and the caller's
 * assertion holds.
 * Expected: VERIFICATION SUCCESSFUL
 */
typedef struct { int received; } S;

void f(S *self, int v)
{
    __ESBMC_requires(self != 0);
    __ESBMC_ensures(self->received == v);
    self->received = v;
}

int main(void)
{
    S s;
    s.received = 0;
    f(&s, 7);
    __ESBMC_assert(s.received == 7, "ensures pins received to v");
    return 0;
}
