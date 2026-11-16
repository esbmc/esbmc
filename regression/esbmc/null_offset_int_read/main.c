struct dev {
  char pad[432];
  void *priv;
};

struct dev d;
_Bool nondet_bool(void);

void *detach(struct dev *dev) {
  unsigned long a = (unsigned long)dev;
  unsigned long b = a + 432;
  return *((void **)b);
}

int main() {
  struct dev *p = nondet_bool() ? &d : 0;
  if (p)
    return detach(p) != 0;
  return 0;
}
