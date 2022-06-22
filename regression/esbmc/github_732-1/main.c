typedef union {
  struct {
    int
      x: 12,
      y: (52ULL - 44);
  };
  int z;
} s1;

typedef union {
  int x;
} s2;

typedef struct {
  s1 x;
} s3;

typedef struct __attribute__((__packed__)) {
  s2 x;
  char y;
  s3 z;
} s4;

int main() {
  s1 v1;
  v1.z = 0;
  v1.x = 0;

  s4 v4;
  s4* v4p = &v4;
  v4p->x.x = v1.z;

  if (v1.z != v4p->z.x.z) {
    v4p->z.x.z = v1.z;
  }

  return 0;
}
