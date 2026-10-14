// --dead-code-check reports only on the base-case pass, so combining it with a
// standalone phase mode (--forward-condition here) would exit without ever
// printing the findings. The combination must be rejected up front.
int main(void)
{
  return 0;
}
