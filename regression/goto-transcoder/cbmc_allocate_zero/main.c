/* __CPROVER_allocate(n, 1) returns a zero-filled object: CBMC's calloc and
 * its DFCC contracts library allocate this way. */
void *__CPROVER_allocate(__CPROVER_size_t, __CPROVER_bool);
int main(void)
{
  int *p = __CPROVER_allocate(4 * sizeof(int), 1);
  __CPROVER_assert(p[3] == 0, "the object is zero-filled");
  return 0;
}
