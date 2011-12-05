
_Bool nondet_bool();
float nondet_float();

int main()
{
  
  float a, b, c;

  a = nondet_float();
  b = nondet_float();

  if (nondet_bool())
    c = a + b;
  else
    c = a - b;

  assert(c==a+b || c==a-b);
}
