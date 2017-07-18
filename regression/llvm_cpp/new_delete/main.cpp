struct X 
{ 
  X(){}
  ~X(){}
};

int main(void)
{
  int *i = new int(1);
  struct X *x = new struct X;

  delete x;
  delete i;

  int *i1 = new int[2];
  struct X *x1 = new struct X[2];

  delete[] i1;
  delete[] x1;

  return 0;
}
