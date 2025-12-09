_Bool a = 1;
_Bool b = 0;

int main() {
  goto c;
d:
  __ESBMC_assert(b, "");
c:
  b = a;
  a = 0;
  goto d;
}
