// Building the counterexample read the handler back from the table and then
// queried the function itself, aborting with "Unimplemented type'd expression
// (5) in smt get".
int nondet_int(void);

void (*handlers[6])(int);

void handler(int sig)
{
  (void)sig;
}

int main(void)
{
  int installed = nondet_int();
  __ESBMC_assume(installed >= 0 && installed < 6);
  handlers[installed] = handler;

  int raised = nondet_int();
  __ESBMC_assume(raised >= 0 && raised < 6);
  void (*resolved)(int) = handlers[raised];
  __ESBMC_assert(resolved == 0, "no handler installed for the raised slot");
  return 0;
}
