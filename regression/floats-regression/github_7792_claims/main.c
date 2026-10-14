// #7792: goto_check dedups claims through std::set<expr2tc>, so the ordering
// must keep literals that differ only in the sign of zero apart.
int main(void)
{
  float f;
  int a = (int)(f + 0.0f) + (int)(f + -0.0f);
  return a;
}
