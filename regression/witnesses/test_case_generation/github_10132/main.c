// #10132: an undefined extern is havocked, but no harness feeds it.
extern void reach_error(void);
extern int __VERIFIER_nondet_int(void);
extern int sys_nerr;

int main(void)
{
  int x = __VERIFIER_nondet_int();
  if (x == 42)
    reach_error();
  return 0;
}
