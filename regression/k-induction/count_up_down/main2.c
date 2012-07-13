
unsigned int nondet_uint();

int main(int argc, char **argv)
{
  unsigned int n = nondet_uint();
  unsigned int x=n, y=0;
  assert(y+x==n);
  while(x>0){
    __ESBMC_assume(y+x==n);
    x--;
    y++;
    assert(y+x==n);
  }
  assert(y==n);

  return 0;
}
