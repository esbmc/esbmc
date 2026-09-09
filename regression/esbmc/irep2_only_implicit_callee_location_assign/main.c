/* The same call on an assignment's right-hand side. The statement here is an
   expression statement whose operand is the assignment, so the pre-recursion
   path sees the assignment rather than the call: the location has to come off
   the sideeffect2t. */
int main(void)
{
  int x;
  x = undeclared_fn(1);
  return x;
}
