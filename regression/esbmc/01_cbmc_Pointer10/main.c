struct S
{
  int x;
  int array[10];
  int z;
};

int main()
{
  struct S s;
  int *p;

  p=&(s.array[4]);
  *p=5;
  
  assert(s.array[4]==5);
  
  p=&s.z;
  *p=6;
  
  assert(s.z==6);
}
