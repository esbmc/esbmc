int roundInt(double a) {
  if (a>=0)
    return (int)(a+0.5);
  else
    return (int)(a-0.5);
}

void main() {
  double a = -0.6;
  assert(roundInt(a) == -1);
}

