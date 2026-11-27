/* #11111: CBMC's __CPROVER_allocate never returns NULL, whatever its
 * zero-init flag, and without --force-malloc-success. */
void *__CPROVER_allocate(__CPROVER_size_t, __CPROVER_bool);
int main(void)
{
  int *p = __CPROVER_allocate(sizeof(int), 0);
  int *q = __CPROVER_allocate(2 * sizeof(int), 1);
  __CPROVER_assert(p != 0, "allocate never returns NULL");
  __CPROVER_assert(q != 0 && q[1] == 0, "a zero-filled allocation is not NULL");
  return 0;
}
