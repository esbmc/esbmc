// &s[i].f names a member of s[i], so s[i] must exist: i == 2 is out of range
// even though &s[2] alone would be valid.
struct pair
{
  int f, g;
} s[2];
int main()
{
  int i = 2;
  int *p = &s[i].f;
  return 0;
}
