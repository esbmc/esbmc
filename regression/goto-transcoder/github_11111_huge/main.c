/* #11111: a request above PTRDIFF_MAX still yields NULL rather than an
 * assumption that prunes every later path. */
void *__CPROVER_allocate(__CPROVER_size_t, __CPROVER_bool);
int main(void)
{
  char *p = __CPROVER_allocate((__CPROVER_size_t)-1, 0);
  __CPROVER_assert(p != 0, "an oversized request is not assumed away");
  return 0;
}
