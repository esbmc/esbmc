// #9373: pb->len reads B::len (8) of the new C, so the loop overruns buf.
struct V { int v = 0; };
struct A { int a = 1; };
struct B { int len = 8; };
struct C : A, B, virtual V { };

int main()
{
  B *pb = new C;
  int buf[4] = {0};
  for (int i = 0; i < pb->len; i++)
    buf[i] = i;
  return buf[0];
}
