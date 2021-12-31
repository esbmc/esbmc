struct MixedData
{
  char var1; // i8
  short var2; // i16
  int var3; // i32
  char var4; // i8
};

int main() {
  struct MixedData a; // total: 96
  assert(sizeof(a) == 12);
  return 0;
}