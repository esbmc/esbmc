
int main()
{
  int *p=malloc(sizeof(int));
  int x;
  int i;
  
  if(i==4711) p=&x;

  // should fail if i==4711
  free(p);
}
