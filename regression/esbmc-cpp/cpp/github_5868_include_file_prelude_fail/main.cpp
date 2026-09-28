// github #5868 gap 5, negative direction: the forced header is really compiled
// and its helper really returns 3, so a wrong expectation must FAIL. Otherwise
// the positive test could pass without the header contributing anything.
int main()
{
  assert(helper() == 4);
  return 0;
}
