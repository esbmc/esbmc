/* The local a pointer value lands in is havocked. */
void __CPROVER_havoc_object(void *);
_Bool nondet_bool(void);

int main(void)
{
  int x = 5, y = 6;
  int *xy = nondet_bool() ? &x : &y;
  __CPROVER_havoc_object(xy);
  __CPROVER_assert(x == 5 && y == 6, "both locals kept");
  return 0;
}
