/* #11111: the allocation is not NULL, so the out-of-bounds write is the
 * defect reported. */
void *__CPROVER_allocate(__CPROVER_size_t, __CPROVER_bool);
int main(void)
{
  int *p = __CPROVER_allocate(sizeof(int), 0);
  p[1] = 0;
  return 0;
}
