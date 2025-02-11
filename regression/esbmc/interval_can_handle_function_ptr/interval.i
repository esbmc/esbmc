int c, e = 1;

int a();
void b();
void d();

int main()
{
  // e = [1,1], c = [0,0]
  c = 1;
  if (c == 1) {
    b();
    // e = [2,2]
    (*a)();
    // clear everything
  }
  else
  {
    e = 1;
  }
  
  d();
}
void b() { 
  int tmp = e;
  e = tmp + 1;
}
void d() {
  if (e != 1)
    __ESBMC_assert(0, "error");
}
