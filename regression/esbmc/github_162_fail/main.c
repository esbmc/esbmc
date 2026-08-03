/* SUCCESSFUL is correct despite the name: b is a zero-initialised global, so
   the addition is 0 + a and cannot overflow. compound_assign_narrow_overflow
   pins the overflow this was reaching for (#6589). */
union {
  int a : 3
} b;

int main()
{
  int a;
  b.a += a;
}
