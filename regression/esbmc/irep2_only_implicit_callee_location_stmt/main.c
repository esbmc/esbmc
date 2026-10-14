/* Nested expression statements: the walk restores the enclosing statement's
   location on the way out, so a call in a loop body does not inherit the one
   belonging to a statement it is not inside. */
int main(void)
{
  int a = 0;

  outer(a);

  while (a)
  {
    inner(a);
    a = 0;
  }

  return 0;
}
