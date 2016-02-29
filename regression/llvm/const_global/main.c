struct X { int a; };

typedef const _Bool inteiro;

const unsigned int d = 1;
const unsigned int c = d;

int fun4(int x, int y) { return x/c; }

int main()
{
  int x;
  x = fun4(1,0);
}
