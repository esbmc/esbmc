/* CBMC emits the <stdlib.h> string-to-integer parsers as bodyless
   FUNCTION_CALL externals; without the libc-body bridge ESBMC returns nondet
   and a valid atoi("42")==42 reports FAILED where CBMC says SUCCESSFUL.
   CBMC 6.8.0 models atoi/atol/strtol (not atoll/strtoll). */
extern int atoi(const char *);
extern long atol(const char *);
extern long strtol(const char *, char **, int);

int main(void)
{
  __CPROVER_assert(atoi("42") == 42, "atoi positive");
  __CPROVER_assert(atoi("-7") == -7, "atoi negative");
  __CPROVER_assert(atoi("0") == 0, "atoi zero");
  __CPROVER_assert(atol("123") == 123L, "atol");
  __CPROVER_assert(strtol("55", 0, 10) == 55L, "strtol base 10");
  __CPROVER_assert(strtol("ff", 0, 16) == 255L, "strtol base 16");
  return 0;
}
