template <class T>
struct vector
{
  T buf;
};
struct b;
struct c
{
  virtual b *e();
};
struct b : c
{
  b *e();
};
vector<c> fg;

int main()
{
}