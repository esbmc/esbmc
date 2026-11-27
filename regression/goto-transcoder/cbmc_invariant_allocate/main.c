/* The adapter cannot read an allocate whose zero-init flag is not a literal,
 * so it drops the invariant and the loop is unwound. */
void *__CPROVER_allocate(__CPROVER_size_t, _Bool);
_Bool nondet_bool(void);
int a[4];

int main(void)
{
  _Bool zero = nondet_bool();
  for (unsigned i = 0; i < 4; i++)
    __CPROVER_loop_invariant(i <= 4 && __CPROVER_allocate(4, zero) != 0)
  {
    a[i] = 1;
  }
  __CPROVER_assert(a[3] == 1, "loop wrote a[3]");
  return 0;
}
