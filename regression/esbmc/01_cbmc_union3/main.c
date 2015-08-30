int my_func(int stat_loc)
{
  // this is a 'temporary union'
  return ((union { __typeof(stat_loc) __in; int __i; })
    { .__in =(stat_loc) }).__i;
}

int main(void)
{
  int x;
  assert(my_func(x)==x);
} 
