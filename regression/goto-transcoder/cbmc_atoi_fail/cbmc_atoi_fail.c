/* Negative variant: atoi is really parsed, not vacuously nondet. */
extern int atoi(const char *);

int main(void)
{
  __CPROVER_assert(atoi("42") == 43, "wrong parse");
  return 0;
}
