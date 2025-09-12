struct {
  int a;
} b[];
void c() { b->a; }
int main()
{
  return 0;
}
