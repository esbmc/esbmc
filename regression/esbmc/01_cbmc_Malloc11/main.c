void *malloc(unsigned s);

int main()
{
  int *p;
  
  p=malloc(100*sizeof(int));

  // buffer overflow
  p[100]=1;
}
