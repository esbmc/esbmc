/* `if (1)` lowers to `IF !1 THEN GOTO`, so the guard is a constant hiding
   under a not. Instrumenting both directions leaves one probe unsatisfiable
   by construction, which used to be reported as CWE-561. */
int f(void)
{
  return 1;
}

int main(void)
{
  if (1)
    f();
  return 0;
}
