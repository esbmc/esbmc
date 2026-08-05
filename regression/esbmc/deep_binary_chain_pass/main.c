/* 3 * 4^6 = 12288 operands. Conversion recurses once per operand, which used
 * to overflow the stack before reaching symbolic execution, see #6617. */
#define A1 x && x && x && x
#define A2 A1 && A1 && A1 && A1
#define A3 A2 && A2 && A2 && A2
#define A4 A3 && A3 && A3 && A3
#define A5 A4 && A4 && A4 && A4
#define A6 A5 && A5 && A5 && A5
#define DEEP A6 && A6 && A6

int deep(int x)
{
  return DEEP;
}

int main(void)
{
  return 0;
}
