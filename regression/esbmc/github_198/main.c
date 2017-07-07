#define balh 1

int main()
{
  unsigned int SIZE = 10;
  int a[SIZE][SIZE][SIZE]; 
  a[SIZE / 2][SIZE-1][SIZE-1] = 1;
  assert(a[SIZE / 2][SIZE-1][SIZE-1] == 1);
}

