/* #7521 field-sensitivity corpus: the #7521 reproducer, a field written once and read in a loop.
 * The struct has 512 fields; F<n>(p) declares n padding fields. */
#include <assert.h>

#define F1(p) int p;
#define F2(p) F1(p##0) F1(p##1)
#define F4(p) F2(p##0) F2(p##1)
#define F8(p) F4(p##0) F4(p##1)
#define F16(p) F8(p##0) F8(p##1)
#define F32(p) F16(p##0) F16(p##1)
#define F64(p) F32(p##0) F32(p##1)
#define F128(p) F64(p##0) F64(p##1)
#define F256(p) F128(p##0) F128(p##1)
struct temp
{
  int x, a1, a2, a3, a4, a5, a6, a7;
  F256(p) F128(q) F64(r) F32(s) F16(t) F8(u)
};

int fun()
{
  struct temp temp1;
  temp1.x = 1;
  for (int i = 0; i < 20; i++)
    for (int j = 0; j < 10; j++)
      if (temp1.x)
        assert(j < 9);
  return 0;
}

int main()
{
  for (int k = 0; k < 100; k++)
    fun();
  return 0;
}
