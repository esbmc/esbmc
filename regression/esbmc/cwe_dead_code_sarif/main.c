extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int x = __VERIFIER_nondet_int();
  __ESBMC_assume(x > 10);
  // Dead else branch (see cwe_dead_code); this variant checks the SARIF
  // advisory (result.level == "note", CWE-561 taxon).
  if (x > 5)
    return 0;
  else
    return 1;
}
