_Bool nondet_bool();
void *malloc(unsigned s);

int analyze_this()
{
  char *p_char=malloc(sizeof(char));
  int *p_int=malloc(sizeof(int));
  void *p;
  
  p=nondet_bool()?p_char:p_int;

  p_int=p;
  free(p_int);
  // this should fail, as it's the wrong type
  *p_int=1;
}

int main()
{
  analyze_this();
}
