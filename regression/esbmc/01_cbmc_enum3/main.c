#if 0
typedef enum ts { A, B } TS;

int main(void)
{
  TS token;

  if(token!=A) token=B;
  
  assert(token==A);

  return 1;
}
#else

enum ts { Ax, Bx, Cx=(Bx<<1)>>1 };

int main(void)
{
  enum ts token;

  if(token!=Bx) token=Bx;

  assert(token==Cx);

  return 1;
} 

#endif
