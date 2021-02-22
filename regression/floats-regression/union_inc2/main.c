  union {
    float a;
    int b;
  } from_union = { .a = 1.0f };

int main(void)
{
  from_union.b++;
  float a = from_union.a;
  assert(from_union.a == 0x1.000002p+0);
}

