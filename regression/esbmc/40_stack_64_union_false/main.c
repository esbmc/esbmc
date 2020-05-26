union MixedData
{
  char var1; // i8
  short var2; // i16
  int var3; // i32
  char var4; // i8
};

int main() {
  union MixedData a; // total: 32
  return 0;
}