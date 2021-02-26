static int top=0;

int get_top(void)
{
  return top;
}

int main(void) 
{
  if (get_top()==0) 
  {
    return 0;
  } 

  assert(0);
  return 0;
}
