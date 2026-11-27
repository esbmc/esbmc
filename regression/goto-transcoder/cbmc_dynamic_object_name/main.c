/* A variable may be called is_dynamic_object, as one in CBMC's contracts
 * library is. */
int main(void)
{
  _Bool is_dynamic_object = 1;
  __CPROVER_assert(is_dynamic_object, "a variable, not the primitive");
  return 0;
}
