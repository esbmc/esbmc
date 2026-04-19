_Bool a = 1;
_Bool b = 0;

int main() {
  goto c;
d:
  b = a;
c:
  b = a;
  a = 0;
  __ESBMC_assert(b, "");
  goto d;
}
