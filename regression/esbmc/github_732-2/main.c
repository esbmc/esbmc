typedef union {
  int x;
} s1;

typedef union {
  struct {
    int x;
  };
  int y;
} s2;

typedef struct {
  s1 x;
} s3;

typedef struct __attribute((__packed__)){
  s2 x;
  char y;
  s3 z;
} s4;

typedef struct {
  char x[11712];
} s5 ;

typedef struct {
  union {
#ifndef RANGE_CHECK_ERROR
    s5 x;
#endif
  };
} s6;

typedef struct {
  s4 x;
  s6 y;
} s7;

s7 gv7;

int main() {
  s1 v1;
  s4* v4 = &gv7.x;

  v1.x = 0;
  v4->x.x = v1.x;

  if (v1.x != v4->z.x.x) {
    v4->z.x.x = v1.x;
  }

  return 0;
}
