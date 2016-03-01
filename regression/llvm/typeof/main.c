#include <assert.h>

int A() { return 1; }
int B() { return 2; }
int C() { return 3; }

#define pointer(T) typeof(T *)
#define array(T, N) typeof(T [N])


struct s { int a; } __attribute__((deprecated)) x;

typeof(x) y;

union un{ int a; } __attribute__((deprecated)) u;

typeof(     u) z;

enum E{ one} __attribute__((deprecated))  e;

typeof( e) w;

struct foo { int x; } __attribute__((deprecated));
typedef struct foo bar __attribute__((deprecated));
bar x1;

struct gorf { int x; };
typedef struct gorf T __attribute__((deprecated));
T t;
void wee() { typeof(t) y; }

typedef __typeof(sizeof(int)) size_t;

struct {unsigned x : 2;} x3;
__typeof__((x3.x+=1)+1) y2;
__typeof__(x3.x<<1) y3;

struct { int x : 8; } x6;
long long y1;
__typeof__(((long long)x1.x + 1)) y1;

enum E1 { ec1, ec2, ec3 };
struct S {
  enum E1         e : 2;
  unsigned short us : 4;
  unsigned long long ul1 : 8;
  unsigned long long ul2 : 50;
} s;

__typeof(s.e + s.e) x_e;
int x_e;

__typeof(s.us + s.us) x_us;
int x_us;

__typeof(s.ul1 + s.ul1) x_ul1;
int x_ul1;

__typeof(s.ul2 + s.ul2) x_ul2;
unsigned long long x_ul2;

__typeof(+(_Bool)0) should_be_int;
int should_be_int;

void *h0(unsigned a0,     ...);
__typeof (h0) h1 __attribute__((__sentinel__));
__typeof (h1) h1 __attribute__((__sentinel__));

// PR3840
void i0 (unsigned short a0);
__typeof (i0) i1;
__typeof (i1) i1;

void PotentiallyEvaluatedTypeofWarn(int n) {
  __typeof(*(0 << 32,(int(*)[n])0)) x;
  (void)x;
}

void f(double a[restrict][5]) { __typeof(a) x = 10; }

unsigned y4;
__typeof(1+1u) y4;
__typeof(1u+1) y4;

long long z1;
__typeof(1ll+1u) z1;

__typeof__(2147483648) x32;
__typeof__(2147483648l) x32;
__typeof__(2147483648L) x32;

typedef __typeof((int*) 0 - (int*) 0) intptr_t;

int foo2(int x, int y) { return x + y; }
typeof(foo2) bar1;
int bar1(int x, int y) { return x + y; }

int main()
{
  typeof(({ unsigned long __ptr; (int *)(0); })) __val1;

  struct Bug {
    typeof(({ unsigned long __ptr; (int *)(0); })) __val;
    union Nested {
      typeof(({ unsigned long __ptr; (int *)(0); })) __val;
    } n;
  } x2;

  typeof(x1) y6;

  int (*x[3]) () = {A, B, C};

  typeof (x[0](1)) a;
  typeof (*x) y;
  typeof (*x) y1[3] = {A, B, C};

  typeof (int) y4;
  typeof (1) y5;

  typeof (typeof (char *)[4]) y2;
  array (pointer (char), 4) y3;

  return 0;
}
