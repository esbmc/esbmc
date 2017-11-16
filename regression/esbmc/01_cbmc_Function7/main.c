typedef unsigned B[8];

void fun(B b)
{
}

int main(void)
{
  B var;
  unsigned *p = &var;
  fun(p);
}
      
