int c, e = 1;

void b();
void d();

int main()
{
  // e = [1,1], c = [0,0]
  c = 1;
  if (c == 1)
    b();
  
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
