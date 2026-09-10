/* The declaration main() sees has an empty parameter list, so the call carries
   no operand for the parameter the definition declares. */
void f();

int main(void)
{
  f();
  return 0;
}

void f(int a)
{
}
