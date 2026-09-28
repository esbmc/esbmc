/* Negative variant: toupper is really computed, not vacuously nondet. */
extern int toupper(int);

int main(void)
{
  __CPROVER_assert(toupper('a') == 'B', "wrong upper-case mapping");
  return 0;
}
