int main(void)
{
  int a[4];
  __CPROVER_size_t k = 1;
  /* CBMC serialises OBJECT_SIZE under the +, which is the shape that used to
     abort migration before the call could be lifted to statement level. */
  __CPROVER_assert(k + __CPROVER_OBJECT_SIZE(&a[0]) == 18, "nested size fail");
  return 0;
}
