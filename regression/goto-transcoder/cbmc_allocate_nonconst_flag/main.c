/* CBMC itself rejects a zero-init flag that is not a constant. */
void *__CPROVER_allocate(__CPROVER_size_t, __CPROVER_bool);
int main(void)
{
  int zero;
  int *p = __CPROVER_allocate(sizeof(int), zero);
  return 0;
}
