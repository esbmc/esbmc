// &f[7] points at the last element, so q[1] is out of bounds.
struct Cols
{
  const unsigned short *ptr_;
  const unsigned short &operator[](long i) const
  {
    return ptr_[i];
  }
};

int main()
{
  unsigned short frac[8] = {0};
  Cols f{frac};
  const unsigned short *q = &f[7];
  return q[1];
}
