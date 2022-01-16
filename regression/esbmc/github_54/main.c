# 1 "input.c"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "input.c"
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 1
# 18 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h"
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/definitions.h" 1
# 121 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/definitions.h"
int X_SIZE_VALUE = 0;
int overflow_mode = 1;
int rounding_mode = 0;
# 144 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/definitions.h"
typedef struct
{
  double a[100];
  int a_size;
  double b[100];
  int b_size;
  double sample_time;
  double a_uncertainty[100];
  double b_uncertainty[100];
} digital_system;

typedef struct
{
  double A[4][4];
  double B[4][4];
  double C[4][4];
  double D[4][4];
  double states[4][4];
  double outputs[4][4];
  double inputs[4][4];
  double K[4][4];
  unsigned int nStates;
  unsigned int nInputs;
  unsigned int nOutputs;
} digital_system_state_space;

typedef struct
{
  int int_bits;
  int frac_bits;
  double max;
  double min;
  int default_realization;
  double delta;
  int scale;
  double max_error;
} implementation;

typedef struct
{
  int push;
  int in;
  int sbiw;
  int cli;
  int out;
  int std;
  int ldd;
  int subi;
  int sbci;
  int lsl;
  int rol;
  int add;
  int adc;
  int adiw;
  int rjmp;
  int mov;
  int sbc;
  int ld;
  int rcall;
  int cp;
  int cpc;
  int ldi;
  int brge;
  int pop;
  int ret;
  int st;
  int brlt;
  int cpi;
} instructions;

typedef struct
{
  long clock;
  int device;
  double cycle;
  instructions assembly;
} hardware;
# 19 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/compatibility.h" 1
# 17 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/compatibility.h"
# 1 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 1 3
# 15 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
# 1 "c:\\tools\\mingw\\mingw-0.6.2\\include\\_mingw.h" 1 3
# 41 "c:\\tools\\mingw\\mingw-0.6.2\\include\\_mingw.h" 3

# 42 "c:\\tools\\mingw\\mingw-0.6.2\\include\\_mingw.h" 3
# 16 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 2 3

# 1 "c:\\tools\\mingw\\mingw-0.6.2\\lib\\gcc\\mingw32\\4.9.3\\include\\stddef.h" 1 3 4
# 212 "c:\\tools\\mingw\\mingw-0.6.2\\lib\\gcc\\mingw32\\4.9.3\\include\\stddef.h" 3 4
# 324 "c:\\tools\\mingw\\mingw-0.6.2\\lib\\gcc\\mingw32\\4.9.3\\include\\stddef.h" 3 4
typedef short unsigned int wchar_t;
# 23 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 2 3
# 60 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3

# 70 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
extern int _argc;
extern char **_argv;

extern int *__attribute__((__cdecl__)) __attribute__((__nothrow__))
__p___argc(void);
extern char ***__attribute__((__cdecl__)) __attribute__((__nothrow__))
__p___argv(void);
extern wchar_t ***__attribute__((__cdecl__)) __attribute__((__nothrow__))
__p___wargv(void);
# 111 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
extern __attribute__((__dllimport__)) int __mb_cur_max;
# 136 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
int *__attribute__((__cdecl__)) __attribute__((__nothrow__)) _errno(void);

int *__attribute__((__cdecl__)) __attribute__((__nothrow__)) __doserrno(void);
# 148 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
extern char ***__attribute__((__cdecl__)) __attribute__((__nothrow__))
__p__environ(void);
extern wchar_t ***__attribute__((__cdecl__)) __attribute__((__nothrow__))
__p__wenviron(void);
# 171 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
extern __attribute__((__dllimport__)) int _sys_nerr;
# 195 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
extern __attribute__((__dllimport__)) char *_sys_errlist[];
# 208 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
extern unsigned __attribute__((__cdecl__)) __attribute__((__nothrow__)) int *
__p__osver(void);
extern unsigned __attribute__((__cdecl__)) __attribute__((__nothrow__)) int *
__p__winver(void);
extern unsigned __attribute__((__cdecl__)) __attribute__((__nothrow__)) int *
__p__winmajor(void);
extern unsigned __attribute__((__cdecl__)) __attribute__((__nothrow__)) int *
__p__winminor(void);

extern __attribute__((__dllimport__)) unsigned int _osver;
extern __attribute__((__dllimport__)) unsigned int _winver;
extern __attribute__((__dllimport__)) unsigned int _winmajor;
extern __attribute__((__dllimport__)) unsigned int _winminor;
# 259 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
char **__attribute__((__cdecl__)) __attribute__((__nothrow__))
__p__pgmptr(void);

wchar_t **__attribute__((__cdecl__)) __attribute__((__nothrow__))
__p__wpgmptr(void);
# 292 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
extern __attribute__((__dllimport__)) int _fmode;
# 302 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
double __attribute__((__cdecl__)) __attribute__((__nothrow__))
atof(const char *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) atoi(const char *);
long __attribute__((__cdecl__)) __attribute__((__nothrow__)) atol(const char *);

double __attribute__((__cdecl__)) __attribute__((__nothrow__))
_wtof(const wchar_t *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_wtoi(const wchar_t *);
long __attribute__((__cdecl__)) __attribute__((__nothrow__))
_wtol(const wchar_t *);

double __attribute__((__cdecl__)) __attribute__((__nothrow__))
__strtod(const char *, char **);
extern double __attribute__((__cdecl__)) __attribute__((__nothrow__))
strtod(const char *__restrict__ __nptr, char **__restrict__ __endptr);
float __attribute__((__cdecl__)) __attribute__((__nothrow__))
strtof(const char *__restrict__, char **__restrict__);
long double __attribute__((__cdecl__)) __attribute__((__nothrow__))
strtold(const char *__restrict__, char **__restrict__);

long __attribute__((__cdecl__)) __attribute__((__nothrow__))
strtol(const char *, char **, int);
unsigned long __attribute__((__cdecl__)) __attribute__((__nothrow__))
strtoul(const char *, char **, int);

long __attribute__((__cdecl__)) __attribute__((__nothrow__))
wcstol(const wchar_t *, wchar_t **, int);
unsigned long __attribute__((__cdecl__)) __attribute__((__nothrow__))
wcstoul(const wchar_t *, wchar_t **, int);
double __attribute__((__cdecl__)) __attribute__((__nothrow__))
wcstod(const wchar_t *, wchar_t **);

float __attribute__((__cdecl__)) __attribute__((__nothrow__))
wcstof(const wchar_t *__restrict__, wchar_t **__restrict__);
long double __attribute__((__cdecl__)) __attribute__((__nothrow__))
wcstold(const wchar_t *__restrict__, wchar_t **__restrict__);

wchar_t *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_wgetenv(const wchar_t *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_wputenv(const wchar_t *);
void __attribute__((__cdecl__)) __attribute__((__nothrow__))
_wsearchenv(const wchar_t *, const wchar_t *, wchar_t *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_wsystem(const wchar_t *);
void __attribute__((__cdecl__)) __attribute__((__nothrow__)) _wmakepath(
  wchar_t *,
  const wchar_t *,
  const wchar_t *,
  const wchar_t *,
  const wchar_t *);
void __attribute__((__cdecl__)) __attribute__((__nothrow__))
_wsplitpath(const wchar_t *, wchar_t *, wchar_t *, wchar_t *, wchar_t *);
wchar_t *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_wfullpath(wchar_t *, const wchar_t *, size_t);

size_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
wcstombs(char *, const wchar_t *, size_t);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
wctomb(char *, wchar_t);

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
mblen(const char *, size_t);
size_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
mbstowcs(wchar_t *, const char *, size_t);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
mbtowc(wchar_t *, const char *, size_t);

int __attribute__((__cdecl__)) __attribute__((__nothrow__)) rand(void);
void __attribute__((__cdecl__)) __attribute__((__nothrow__))
srand(unsigned int);

void *__attribute__((__cdecl__)) __attribute__((__nothrow__))
calloc(size_t, size_t) __attribute__((__malloc__));
void *__attribute__((__cdecl__)) __attribute__((__nothrow__)) malloc(size_t)
  __attribute__((__malloc__));
void *__attribute__((__cdecl__)) __attribute__((__nothrow__))
realloc(void *, size_t);
void __attribute__((__cdecl__)) __attribute__((__nothrow__)) free(void *);
void __attribute__((__cdecl__)) __attribute__((__nothrow__)) abort(void)
  __attribute__((__noreturn__));
void __attribute__((__cdecl__)) __attribute__((__nothrow__)) exit(int)
  __attribute__((__noreturn__));

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
atexit(void (*)(void));

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
system(const char *);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
getenv(const char *);

void *__attribute__((__cdecl__)) bsearch(
  const void *,
  const void *,
  size_t,
  size_t,
  int (*)(const void *, const void *));
void __attribute__((__cdecl__))
qsort(void *, size_t, size_t, int (*)(const void *, const void *));

int __attribute__((__cdecl__)) __attribute__((__nothrow__)) abs(int)
  __attribute__((__const__));
long __attribute__((__cdecl__)) __attribute__((__nothrow__)) labs(long)
  __attribute__((__const__));
# 384 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
typedef struct
{
  int quot, rem;
} div_t;
typedef struct
{
  long quot, rem;
} ldiv_t;

div_t __attribute__((__cdecl__)) __attribute__((__nothrow__)) div(int, int)
  __attribute__((__const__));
ldiv_t __attribute__((__cdecl__)) __attribute__((__nothrow__)) ldiv(long, long)
  __attribute__((__const__));

void __attribute__((__cdecl__)) __attribute__((__nothrow__))
_beep(unsigned int, unsigned int) __attribute__((__deprecated__));

void __attribute__((__cdecl__)) __attribute__((__nothrow__)) _seterrormode(int)
  __attribute__((__deprecated__));
void __attribute__((__cdecl__)) __attribute__((__nothrow__))
_sleep(unsigned long) __attribute__((__deprecated__));

void __attribute__((__cdecl__)) __attribute__((__nothrow__)) _exit(int)
  __attribute__((__noreturn__));

typedef int (*_onexit_t)(void);
_onexit_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
_onexit(_onexit_t);

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_putenv(const char *);
void __attribute__((__cdecl__)) __attribute__((__nothrow__))
_searchenv(const char *, const char *, char *);

char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_ecvt(double, int, int *, int *);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_fcvt(double, int, int *, int *);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_gcvt(double, int, char *);

void __attribute__((__cdecl__)) __attribute__((__nothrow__))
_makepath(char *, const char *, const char *, const char *, const char *);
void __attribute__((__cdecl__)) __attribute__((__nothrow__))
_splitpath(const char *, char *, char *, char *, char *);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_fullpath(char *, const char *, size_t);

char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_itoa(int, char *, int);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_ltoa(long, char *, int);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_ultoa(unsigned long, char *, int);
wchar_t *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_itow(int, wchar_t *, int);
wchar_t *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_ltow(long, wchar_t *, int);
wchar_t *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_ultow(unsigned long, wchar_t *, int);

long long __attribute__((__cdecl__)) __attribute__((__nothrow__))
_atoi64(const char *);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_i64toa(long long, char *, int);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_ui64toa(unsigned long long, char *, int);
long long __attribute__((__cdecl__)) __attribute__((__nothrow__))
_wtoi64(const wchar_t *);
wchar_t *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_i64tow(long long, wchar_t *, int);
wchar_t *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_ui64tow(unsigned long long, wchar_t *, int);

unsigned int __attribute__((__cdecl__)) __attribute__((__nothrow__)) (
  _rotl)(unsigned int, int)__attribute__((__const__));
unsigned int __attribute__((__cdecl__)) __attribute__((__nothrow__)) (
  _rotr)(unsigned int, int)__attribute__((__const__));
unsigned long __attribute__((__cdecl__)) __attribute__((__nothrow__)) (
  _lrotl)(unsigned long, int)__attribute__((__const__));
unsigned long __attribute__((__cdecl__)) __attribute__((__nothrow__)) (
  _lrotr)(unsigned long, int)__attribute__((__const__));

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_set_error_mode(int);
# 476 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
putenv(const char *);
void __attribute__((__cdecl__)) __attribute__((__nothrow__))
searchenv(const char *, const char *, char *);

char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
itoa(int, char *, int);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
ltoa(long, char *, int);

char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
ecvt(double, int, int *, int *);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
fcvt(double, int, int *, int *);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
gcvt(double, int, char *);
# 496 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
void __attribute__((__cdecl__)) __attribute__((__nothrow__)) _Exit(int)
  __attribute__((__noreturn__));

typedef struct
{
  long long quot, rem;
} lldiv_t;

lldiv_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
lldiv(long long, long long) __attribute__((__const__));

long long __attribute__((__cdecl__)) __attribute__((__nothrow__))
llabs(long long);
# 516 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
long long __attribute__((__cdecl__)) __attribute__((__nothrow__))
strtoll(const char *__restrict__, char **__restrict, int);
unsigned long long __attribute__((__cdecl__)) __attribute__((__nothrow__))
strtoull(const char *__restrict__, char **__restrict__, int);

long long __attribute__((__cdecl__)) __attribute__((__nothrow__))
atoll(const char *);

long long __attribute__((__cdecl__)) __attribute__((__nothrow__))
wtoll(const wchar_t *);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
lltoa(long long, char *, int);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
ulltoa(unsigned long long, char *, int);
wchar_t *__attribute__((__cdecl__)) __attribute__((__nothrow__))
lltow(long long, wchar_t *, int);
wchar_t *__attribute__((__cdecl__)) __attribute__((__nothrow__))
ulltow(unsigned long long, wchar_t *, int);
# 567 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) mkstemp(char *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__mingw_mkstemp(int, char *);
# 609 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
extern __inline__ __attribute__((__always_inline__)) int
  __attribute__((__cdecl__)) __attribute__((__nothrow__))
  mkstemp(char *__filename_template)
{
  return __mingw_mkstemp(0, __filename_template);
}
# 620 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdlib.h" 3
char *__attribute__((__cdecl__)) __attribute__((__nothrow__)) mkdtemp(char *);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
__mingw_mkdtemp(char *);

extern __inline__
  __attribute__((__always_inline__)) char *__attribute__((__cdecl__))
  __attribute__((__nothrow__)) mkdtemp(char *__dirname_template)
{
  return __mingw_mkdtemp(__dirname_template);
}

# 18 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/compatibility.h" 2
# 1 "c:\\tools\\mingw\\mingw-0.6.2\\include\\assert.h" 1 3
# 38 "c:\\tools\\mingw\\mingw-0.6.2\\include\\assert.h" 3
void __attribute__((__cdecl__)) __attribute__((__nothrow__))
_assert(const char *, const char *, int) __attribute__((__noreturn__));
# 19 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/compatibility.h" 2
# 1 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 1 3
# 26 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
# 1 "c:\\tools\\mingw\\mingw-0.6.2\\lib\\gcc\\mingw32\\4.9.3\\include\\stddef.h" 1 3 4
# 353 "c:\\tools\\mingw\\mingw-0.6.2\\lib\\gcc\\mingw32\\4.9.3\\include\\stddef.h" 3 4
typedef short unsigned int wint_t;
# 27 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 2 3

# 1 "c:\\tools\\mingw\\mingw-0.6.2\\lib\\gcc\\mingw32\\4.9.3\\include\\stdarg.h" 1 3 4
# 40 "c:\\tools\\mingw\\mingw-0.6.2\\lib\\gcc\\mingw32\\4.9.3\\include\\stdarg.h" 3 4
typedef __builtin_va_list __gnuc_va_list;
# 29 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 2 3
# 130 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
typedef struct _iobuf
{
  char *_ptr;
  int _cnt;
  char *_base;
  int _flag;
  int _file;
  int _charbuf;
  int _bufsiz;
  char *_tmpfname;
} FILE;
# 155 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
extern __attribute__((__dllimport__)) FILE _iob[];

FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__))
fopen(const char *, const char *);
FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__))
freopen(const char *, const char *, FILE *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) fflush(FILE *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) fclose(FILE *);

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
remove(const char *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
rename(const char *, const char *);
FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__)) tmpfile(void);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__)) tmpnam(char *);

char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_tempnam(const char *, const char *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) _rmtmp(void);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_unlink(const char *);

char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
tempnam(const char *, const char *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) rmtmp(void);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
unlink(const char *);

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
setvbuf(FILE *, char *, int, size_t);

void __attribute__((__cdecl__)) __attribute__((__nothrow__))
setbuf(FILE *, char *);
# 203 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
extern int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__mingw_fprintf(FILE *, const char *, ...);
extern int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__mingw_printf(const char *, ...);
extern int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__mingw_sprintf(char *, const char *, ...);
extern int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__mingw_snprintf(char *, size_t, const char *, ...);
extern int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__mingw_vfprintf(FILE *, const char *, __gnuc_va_list);
extern int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__mingw_vprintf(const char *, __gnuc_va_list);
extern int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__mingw_vsprintf(char *, const char *, __gnuc_va_list);
extern int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__mingw_vsnprintf(char *, size_t, const char *, __gnuc_va_list);
# 292 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
fprintf(FILE *, const char *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
printf(const char *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
sprintf(char *, const char *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
vfprintf(FILE *, const char *, __gnuc_va_list);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
vprintf(const char *, __gnuc_va_list);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
vsprintf(char *, const char *, __gnuc_va_list);
# 307 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__msvcrt_fprintf(FILE *, const char *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__msvcrt_printf(const char *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__msvcrt_sprintf(char *, const char *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__msvcrt_vfprintf(FILE *, const char *, __gnuc_va_list);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__msvcrt_vprintf(const char *, __gnuc_va_list);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
__msvcrt_vsprintf(char *, const char *, __gnuc_va_list);

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_snprintf(char *, size_t, const char *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_vsnprintf(char *, size_t, const char *, __gnuc_va_list);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_vscprintf(const char *, __gnuc_va_list);
# 330 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
snprintf(char *, size_t, const char *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
vsnprintf(char *, size_t, const char *, __gnuc_va_list);

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
vscanf(const char *__restrict__, __gnuc_va_list);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
vfscanf(FILE *__restrict__, const char *__restrict__, __gnuc_va_list);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
vsscanf(const char *__restrict__, const char *__restrict__, __gnuc_va_list);

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
fscanf(FILE *, const char *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
scanf(const char *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
sscanf(const char *, const char *, ...);

int __attribute__((__cdecl__)) __attribute__((__nothrow__)) fgetc(FILE *);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__))
fgets(char *, int, FILE *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) fputc(int, FILE *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
fputs(const char *, FILE *);
char *__attribute__((__cdecl__)) __attribute__((__nothrow__)) gets(char *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) puts(const char *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) ungetc(int, FILE *);

int __attribute__((__cdecl__)) __attribute__((__nothrow__)) _filbuf(FILE *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_flsbuf(int, FILE *);

extern __inline__ int __attribute__((__cdecl__)) __attribute__((__nothrow__))
getc(FILE *__F)
{
  return (--__F->_cnt >= 0) ? (int)(unsigned char)*__F->_ptr++ : _filbuf(__F);
}

extern __inline__ int __attribute__((__cdecl__)) __attribute__((__nothrow__))
putc(int __c, FILE *__F)
{
  return (--__F->_cnt >= 0) ? (int)(unsigned char)(*__F->_ptr++ = (char)__c)
                            : _flsbuf(__c, __F);
}

extern __inline__ int __attribute__((__cdecl__)) __attribute__((__nothrow__))
getchar(void)
{
  return (--(&_iob[0])->_cnt >= 0) ? (int)(unsigned char)*(&_iob[0])->_ptr++
                                   : _filbuf((&_iob[0]));
}

extern __inline__ int __attribute__((__cdecl__)) __attribute__((__nothrow__))
putchar(int __c)
{
  return (--(&_iob[1])->_cnt >= 0)
           ? (int)(unsigned char)(*(&_iob[1])->_ptr++ = (char)__c)
           : _flsbuf(__c, (&_iob[1]));
}
# 411 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
size_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
fread(void *, size_t, size_t, FILE *);
size_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
fwrite(const void *, size_t, size_t, FILE *);

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
fseek(FILE *, long, int);
long __attribute__((__cdecl__)) __attribute__((__nothrow__)) ftell(FILE *);
void __attribute__((__cdecl__)) __attribute__((__nothrow__)) rewind(FILE *);
# 454 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
typedef long long fpos_t;

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
fgetpos(FILE *, fpos_t *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
fsetpos(FILE *, const fpos_t *);

int __attribute__((__cdecl__)) __attribute__((__nothrow__)) feof(FILE *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) ferror(FILE *);
# 479 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
void __attribute__((__cdecl__)) __attribute__((__nothrow__)) clearerr(FILE *);
void __attribute__((__cdecl__)) __attribute__((__nothrow__))
perror(const char *);

FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_popen(const char *, const char *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) _pclose(FILE *);

FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__))
popen(const char *, const char *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) pclose(FILE *);

int __attribute__((__cdecl__)) __attribute__((__nothrow__)) _flushall(void);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) _fgetchar(void);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) _fputchar(int);
FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_fdopen(int, const char *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) _fileno(FILE *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) _fcloseall(void);
FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_fsopen(const char *, const char *, int);

int __attribute__((__cdecl__)) __attribute__((__nothrow__)) _getmaxstdio(void);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) _setmaxstdio(int);
# 526 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
unsigned int __attribute__((__cdecl__)) __mingw_get_output_format(void);
unsigned int __attribute__((__cdecl__)) __mingw_set_output_format(unsigned int);

int __attribute__((__cdecl__)) __mingw_get_printf_count_output(void);
int __attribute__((__cdecl__)) __mingw_set_printf_count_output(int);
# 553 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
extern __inline__ __attribute__((__always_inline__)) unsigned int
  __attribute__((__cdecl__)) _get_output_format(void)
{
  return __mingw_get_output_format();
}

extern __inline__ __attribute__((__always_inline__)) unsigned int
  __attribute__((__cdecl__)) _set_output_format(unsigned int __style)
{
  return __mingw_set_output_format(__style);
}
# 579 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
extern __inline__ __attribute__((__always_inline__)) int
  __attribute__((__cdecl__)) _get_printf_count_output(void)
{
  return 0 ? 1 : __mingw_get_printf_count_output();
}

extern __inline__ __attribute__((__always_inline__)) int
  __attribute__((__cdecl__)) _set_printf_count_output(int __mode)
{
  return 0 ? 1 : __mingw_set_printf_count_output(__mode);
}

int __attribute__((__cdecl__)) __attribute__((__nothrow__)) fgetchar(void);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) fputchar(int);
FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__))
fdopen(int, const char *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) fileno(FILE *);
# 599 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
# 1 "c:\\tools\\mingw\\mingw-0.6.2\\include\\sys\\types.h" 1 3
# 35 "c:\\tools\\mingw\\mingw-0.6.2\\include\\sys\\types.h" 3

# 36 "c:\\tools\\mingw\\mingw-0.6.2\\include\\sys\\types.h" 3
# 45 "c:\\tools\\mingw\\mingw-0.6.2\\include\\sys\\types.h" 3
# 1 "c:\\tools\\mingw\\mingw-0.6.2\\lib\\gcc\\mingw32\\4.9.3\\include\\stddef.h" 1 3 4
# 147 "c:\\tools\\mingw\\mingw-0.6.2\\lib\\gcc\\mingw32\\4.9.3\\include\\stddef.h" 3 4
typedef int ptrdiff_t;
# 46 "c:\\tools\\mingw\\mingw-0.6.2\\include\\sys\\types.h" 2 3

# 1 "c:\\tools\\mingw\\mingw-0.6.2\\include\\parts\\time.h" 1 3
# 64 "c:\\tools\\mingw\\mingw-0.6.2\\include\\parts\\time.h" 3
typedef long __time32_t;
typedef long long __time64_t;
# 74 "c:\\tools\\mingw\\mingw-0.6.2\\include\\parts\\time.h" 3
typedef __time32_t time_t;
# 50 "c:\\tools\\mingw\\mingw-0.6.2\\include\\sys\\types.h" 2 3

typedef long _off_t;

typedef _off_t off_t;

typedef unsigned int _dev_t;

typedef _dev_t dev_t;

typedef short _ino_t;

typedef _ino_t ino_t;

typedef int _pid_t;

typedef _pid_t pid_t;

typedef unsigned short _mode_t;

typedef _mode_t mode_t;

typedef int _sigset_t;

typedef _sigset_t sigset_t;

typedef int _ssize_t;

typedef _ssize_t ssize_t;

typedef long long fpos64_t;

typedef long long off64_t;

typedef unsigned long useconds_t __attribute__((__deprecated__));
# 600 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 2 3
extern __inline__ FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__))
fopen64(const char *filename, const char *mode)
{
  return fopen(filename, mode);
}

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
fseeko64(FILE *, off64_t, int);

extern __inline__ off64_t __attribute__((__cdecl__))
__attribute__((__nothrow__)) ftello64(FILE *stream)
{
  fpos_t pos;
  if(fgetpos(stream, &pos))
    return -1LL;
  else
    return ((off64_t)pos);
}
# 628 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
fwprintf(FILE *, const wchar_t *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
wprintf(const wchar_t *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_snwprintf(wchar_t *, size_t, const wchar_t *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
vfwprintf(FILE *, const wchar_t *, __gnuc_va_list);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
vwprintf(const wchar_t *, __gnuc_va_list);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_vsnwprintf(wchar_t *, size_t, const wchar_t *, __gnuc_va_list);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_vscwprintf(const wchar_t *, __gnuc_va_list);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
fwscanf(FILE *, const wchar_t *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
wscanf(const wchar_t *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
swscanf(const wchar_t *, const wchar_t *, ...);
wint_t __attribute__((__cdecl__)) __attribute__((__nothrow__)) fgetwc(FILE *);
wint_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
fputwc(wchar_t, FILE *);
wint_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
ungetwc(wchar_t, FILE *);

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
swprintf(wchar_t *, const wchar_t *, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
vswprintf(wchar_t *, const wchar_t *, __gnuc_va_list);

wchar_t *__attribute__((__cdecl__)) __attribute__((__nothrow__))
fgetws(wchar_t *, int, FILE *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
fputws(const wchar_t *, FILE *);
wint_t __attribute__((__cdecl__)) __attribute__((__nothrow__)) getwc(FILE *);
wint_t __attribute__((__cdecl__)) __attribute__((__nothrow__)) getwchar(void);
wchar_t *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_getws(wchar_t *);
wint_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
putwc(wint_t, FILE *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_putws(const wchar_t *);
wint_t __attribute__((__cdecl__)) __attribute__((__nothrow__)) putwchar(wint_t);
FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_wfdopen(int, const wchar_t *);
FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_wfopen(const wchar_t *, const wchar_t *);
FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_wfreopen(const wchar_t *, const wchar_t *, FILE *);
FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_wfsopen(const wchar_t *, const wchar_t *, int);
wchar_t *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_wtmpnam(wchar_t *);
wchar_t *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_wtempnam(const wchar_t *, const wchar_t *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_wrename(const wchar_t *, const wchar_t *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
_wremove(const wchar_t *);
void __attribute__((__cdecl__)) __attribute__((__nothrow__))
_wperror(const wchar_t *);
FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__))
_wpopen(const wchar_t *, const wchar_t *);

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
snwprintf(wchar_t *s, size_t n, const wchar_t *format, ...);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
vsnwprintf(wchar_t *s, size_t n, const wchar_t *format, __gnuc_va_list arg);

int __attribute__((__cdecl__)) __attribute__((__nothrow__))
vwscanf(const wchar_t *__restrict__, __gnuc_va_list);
int __attribute__((__cdecl__)) __attribute__((__nothrow__))
vfwscanf(FILE *__restrict__, const wchar_t *__restrict__, __gnuc_va_list);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) vswscanf(
  const wchar_t *__restrict__,
  const wchar_t *__restrict__,
  __gnuc_va_list);
# 690 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdio.h" 3
FILE *__attribute__((__cdecl__)) __attribute__((__nothrow__))
wpopen(const wchar_t *, const wchar_t *);

wint_t __attribute__((__cdecl__)) __attribute__((__nothrow__)) _fgetwchar(void);
wint_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
_fputwchar(wint_t);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) _getw(FILE *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) _putw(int, FILE *);

wint_t __attribute__((__cdecl__)) __attribute__((__nothrow__)) fgetwchar(void);
wint_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
fputwchar(wint_t);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) getw(FILE *);
int __attribute__((__cdecl__)) __attribute__((__nothrow__)) putw(int, FILE *);

# 20 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/compatibility.h" 2

void __DSVERIFIER_assume(_Bool expression)
{
  __ESBMC_assume(expression);
}

void __DSVERIFIER_assert(_Bool expression)
{
  ((expression)
     ? (void)0
     : _assert(
         "expression",
         "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/compatibility.h",
         36));
}

void __DSVERIFIER_assert_msg(_Bool expression, char *msg)
{
  printf("%c", msg);
  ((expression)
     ? (void)0
     : _assert(
         "expression",
         "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/compatibility.h",
         41));
}
# 20 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/fixed-point.h" 1
# 25 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/fixed-point.h"
# 1 "c:\\tools\\mingw\\mingw-0.6.2\\lib\\gcc\\mingw32\\4.9.3\\include\\stdint.h" 1 3 4
# 9 "c:\\tools\\mingw\\mingw-0.6.2\\lib\\gcc\\mingw32\\4.9.3\\include\\stdint.h" 3 4
# 1 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdint.h" 1 3 4
# 24 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdint.h" 3 4
# 1 "c:\\tools\\mingw\\mingw-0.6.2\\lib\\gcc\\mingw32\\4.9.3\\include\\stddef.h" 1 3 4
# 25 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdint.h" 2 3 4

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;

typedef signed char int_least8_t;
typedef unsigned char uint_least8_t;
typedef short int_least16_t;
typedef unsigned short uint_least16_t;
typedef int int_least32_t;
typedef unsigned uint_least32_t;
typedef long long int_least64_t;
typedef unsigned long long uint_least64_t;

typedef signed char int_fast8_t;
typedef unsigned char uint_fast8_t;
typedef short int_fast16_t;
typedef unsigned short uint_fast16_t;
typedef int int_fast32_t;
typedef unsigned int uint_fast32_t;
typedef long long int_fast64_t;
typedef unsigned long long uint_fast64_t;
# 66 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdint.h" 3 4
typedef int intptr_t;
# 75 "c:\\tools\\mingw\\mingw-0.6.2\\include\\stdint.h" 3 4
typedef unsigned int uintptr_t;

typedef long long intmax_t;
typedef unsigned long long uintmax_t;
# 10 "c:\\tools\\mingw\\mingw-0.6.2\\lib\\gcc\\mingw32\\4.9.3\\include\\stdint.h" 2 3 4
# 26 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/fixed-point.h" 2
# 1 "c:\\tools\\mingw\\mingw-0.6.2\\include\\inttypes.h" 1 3
# 9 "c:\\tools\\mingw\\mingw-0.6.2\\include\\inttypes.h" 3
# 1 "c:\\tools\\mingw\\mingw-0.6.2\\lib\\gcc\\mingw32\\4.9.3\\include\\stddef.h" 1 3 4
# 10 "c:\\tools\\mingw\\mingw-0.6.2\\include\\inttypes.h" 2 3

typedef struct
{
  intmax_t quot;
  intmax_t rem;
} imaxdiv_t;
# 256 "c:\\tools\\mingw\\mingw-0.6.2\\include\\inttypes.h" 3
intmax_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
imaxabs(intmax_t j);
# 271 "c:\\tools\\mingw\\mingw-0.6.2\\include\\inttypes.h" 3
imaxdiv_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
imaxdiv(intmax_t numer, intmax_t denom);

intmax_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
strtoimax(const char *__restrict__ nptr, char **__restrict__ endptr, int base);
uintmax_t __attribute__((__cdecl__)) __attribute__((__nothrow__))
strtoumax(const char *__restrict__ nptr, char **__restrict__ endptr, int base);

intmax_t __attribute__((__cdecl__)) __attribute__((__nothrow__)) wcstoimax(
  const wchar_t *__restrict__ nptr,
  wchar_t **__restrict__ endptr,
  int base);
uintmax_t __attribute__((__cdecl__)) __attribute__((__nothrow__)) wcstoumax(
  const wchar_t *__restrict__ nptr,
  wchar_t **__restrict__ endptr,
  int base);

# 27 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/fixed-point.h" 2

extern implementation impl;

typedef int64_t fxp_t;

fxp_t _fxp_one;
fxp_t _fxp_half;
fxp_t _fxp_minus_one;
fxp_t _fxp_min;
fxp_t _fxp_max;

double _dbl_max;
double _dbl_min;

fxp_t _fxp_fmask;

fxp_t _fxp_imask;

static const double scale_factor[31] = {
  1.0,         2.0,        4.0,        8.0,         16.0,        32.0,
  64.0,        128.0,      256.0,      512.0,       1024.0,      2048.0,
  4096.0,      8192.0,     16384.0,    32768.0,     65536.0,     131072.0,
  262144.0,    524288.0,   1048576.0,  2097152.0,   4194304.0,   8388608.0,
  16777216.0,  33554432.0, 67108864.0, 134217728.0, 268435456.0, 536870912.0,
  1073741824.0};

static const double scale_factor_inv[31] = {
  1.0,
  0.5,
  0.25,
  0.125,
  0.0625,
  0.03125,
  0.015625,
  0.0078125,
  0.00390625,
  0.001953125,
  0.0009765625,
  0.00048828125,
  0.000244140625,
  0.0001220703125,
  0.00006103515625,
  0.000030517578125,
  0.000015258789063,
  0.000007629394531,
  0.000003814697266,
  0.000001907348633,
  0.000000953674316,
  0.000000476837158,
  0.000000238418579,
  0.000000119209290,
  0.000000059604645,
  0.000000029802322,
  0.000000014901161,
  0.000000007450581,
  0.000000003725290,
  0.000000001862645,
  0.000000000931323};

static const float rand_uni[10000] = {
  -0.486240329978498f,    -0.0886462298529236f,   -0.140307596103306f,
  0.301096597450952f,     0.0993171079928659f,    0.971751769763271f,
  0.985173975730828f,     0.555993645184930f,     0.582088652691427f,
  -0.153377496651175f,    0.383610009058905f,     -0.335724126391271f,
  0.978768141636516f,     -0.276250018648572f,    0.390075705739569f,
  -0.179022404038782f,    0.690083827115783f,     -0.872530132490992f,
  -0.970585763293203f,    -0.581476053441704f,    -0.532614615674888f,
  -0.239699306693312f,    -0.678183014035494f,    0.349502640932782f,
  -0.210469890686263f,    0.841262085391842f,     -0.473585465151401f,
  0.659383565443701f,     -0.651160036945754f,    -0.961043527561335f,
  -0.0814927639199137f,   0.621303110569702f,     -0.784529166943541f,
  0.0238464770757800f,    0.392694728594110f,     0.776848735202001f,
  0.0870059709310509f,    0.880563655271790f,     0.883457036977564f,
  -0.249235082877382f,    -0.691040749216870f,    0.578731120064320f,
  -0.973932858000832f,    -0.117699105431720f,    -0.723831748151088f,
  -0.483149657477524f,    -0.821277691383664f,    -0.459725618100875f,
  0.148175952221864f,     0.444306875534854f,     -0.325610376336498f,
  0.544142311404910f,     -0.165319440455435f,    0.136706800705517f,
  0.543312481350682f,     0.467210959764607f,     -0.349266618228534f,
  -0.660110730565862f,    0.910332331495431f,     0.961049802789367f,
  -0.786168905164629f,    0.305648402726554f,     0.510815258508885f,
  0.0950733260984060f,    0.173750645487898f,     0.144488668408672f,
  0.0190031984466126f,    -0.299194577636724f,    0.302411647442273f,
  -0.730462524226212f,    0.688646006554796f,     0.134948379722118f,
  0.533716723458894f,     -0.00226300779660438f,  -0.561340777806718f,
  0.450396313744017f,     -0.569445876566955f,    0.954155246557698f,
  -0.255403882430676f,    -0.759820984120828f,    -0.855279790307514f,
  -0.147352581758156f,    -0.302269055643746f,    -0.642038024364086f,
  -0.367405981107491f,    0.491844011712164f,     -0.542191710121194f,
  -0.938294043323732f,    0.683979894338020f,     0.294728290855287f,
  0.00662691839443919f,   -0.931040350582855f,    0.152356209974418f,
  0.678620860551457f,     -0.534989269238408f,    0.932096367913226f,
  -0.0361062818028513f,   -0.847189697149530f,    -0.975903030160255f,
  0.623293205784014f,     -0.661289688031659f,    0.724486055119603f,
  0.307504095172835f,     0.00739266163731767f,   -0.393681596442097f,
  0.0313739422974388f,    0.0768157689673350f,    -0.652063346886817f,
  0.864188030044388f,     -0.588932092781034f,    0.496015896758580f,
  -0.872858269231211f,    0.978780599551039f,     -0.504887732991147f,
  -0.462378791937628f,    0.0141726829338038f,    0.769610007653591f,
  0.945233033188923f,     -0.782235375325016f,    -0.832206533738799f,
  0.745634368088673f,     -0.696969510157151f,    -0.0674631869948374f,
  -0.123186450806584f,    -0.359158959141949f,    -0.393882649464391f,
  0.441371446689899f,     -0.829394270569736f,    -0.301502651277431f,
  -0.996215501187289f,    0.934634037393066f,     -0.282431114746289f,
  -0.927550795619590f,    -0.437037530043415f,    -0.360426812995980f,
  0.949549724575862f,     0.502784616197919f,     0.800771681422909f,
  -0.511398929004089f,    0.309288504642554f,     -0.207261227890933f,
  0.930587995125773f,     -0.777029876696670f,    -0.489329175755640f,
  -0.134595132329858f,    0.285771358983518f,     0.182331373854387f,
  -0.544110494560697f,    0.278439882883985f,     -0.556325158102182f,
  0.579043806545889f,     0.134648133801916f,     0.602850725479294f,
  -0.151663563868883f,    0.180694361855878f,     -0.651591295315595f,
  0.281129147768056f,     -0.580047306475484f,    0.687883075491433f,
  0.279398670804288f,     -0.853428128249503f,    -0.532609367372680f,
  -0.821156786377917f,    -0.181273229058573f,    -0.983898569846882f,
  -0.0964374318311501f,   0.880923372124250f,     0.102643371392389f,
  0.893615387135596f,     -0.259276649383649f,    0.699287743639363f,
  0.402940604635828f,     -0.110721596226581f,    0.0846246472582877f,
  0.820733021865405f,     0.795578903285308f,     -0.495144122011537f,
  0.273150029257472f,     -0.268249949701437f,    0.231982193341980f,
  0.694211299124074f,     0.859950868718233f,     0.959483382623794f,
  -0.422972626833543f,    -0.109621798738360f,    0.433094703426531f,
  0.694025903378851f,     0.374478987547435f,     -0.293668545105608f,
  -0.396213864190828f,    -0.0632095887099047f,   -0.0285139536748673f,
  0.831794132192390f,     -0.548543088139238f,    0.791869201724680f,
  0.325211484201845f,     0.155274810721772f,     -0.112383643064821f,
  -0.674403070297721f,    0.642801068229810f,     -0.615712048835242f,
  -0.322576771285566f,    -0.409336818836595f,    0.548069973193770f,
  -0.386353709407947f,    -0.0741664985357784f,   0.619639599324983f,
  -0.815703814931314f,    0.965550307223862f,     0.623407852683828f,
  -0.789634372832984f,    0.736750050047572f,     -0.0269443926793700f,
  0.00545706093721488f,   -0.315712479832091f,    -0.890110021644720f,
  -0.869390443173846f,    -0.381538869981866f,    -0.109498998005949f,
  0.131433952330613f,     -0.233452413139316f,    0.660289822785465f,
  0.543381186340023f,     -0.384712418750451f,    -0.913477554164890f,
  0.767102957655267f,     -0.115129944521936f,    -0.741161985822647f,
  -0.0604180020782450f,   -0.819131535144059f,    -0.409539679760029f,
  0.574419252943637f,     -0.0440704617157433f,   0.933173744590532f,
  0.261360623390448f,     -0.880290575543046f,    0.329806293425492f,
  0.548915621667952f,     0.635187167795234f,     -0.611034070318967f,
  0.458196727901944f,     0.397377226781023f,     0.711941361933987f,
  0.782147744383368f,     -0.00300685339552631f,  0.384687233450957f,
  0.810102466029521f,     0.452919847968424f,     -0.183164257016897f,
  -0.755603185485427f,    -0.604334477365858f,    -0.786222413488860f,
  -0.434887500763099f,    -0.678845635625581f,    -0.381200370488331f,
  -0.582350534916068f,    -0.0444427346996734f,   0.116237247526397f,
  -0.364680921206275f,    -0.829395404347498f,    -0.258574590032613f,
  -0.910082114298859f,    0.501356900925997f,     0.0295361922006900f,
  -0.471786618165219f,    0.536352925101547f,     -0.316120662284464f,
  -0.168902841718737f,    0.970850119987976f,     -0.813818666854395f,
  -0.0861183123848732f,   0.866784827877161f,     0.535966478165739f,
  -0.806958669103425f,    -0.627307415616045f,    -0.686618354673079f,
  0.0239165685193152f,    0.525427699287402f,     0.834079334357391f,
  -0.527333932295852f,    0.130970034225907f,     -0.790218350377199f,
  0.399338640441987f,     0.133591886379939f,     -0.181354311053254f,
  0.420121912637914f,     -0.625002202728601f,    -0.293296669160307f,
  0.0113819513424340f,    -0.882382002895096f,    -0.883750159690028f,
  0.441583656876336f,     -0.439054135454480f,    0.873049498123622f,
  0.660844523562817f,     0.0104240153103699f,    0.611420248331623f,
  -0.235926309432748f,    0.207317724918460f,     0.884691834560657f,
  0.128302402592277f,     -0.283754448219060f,    0.237649901255856f,
  0.610200763264703f,     -0.625035441247926f,    -0.964609592118695f,
  -0.323146562743113f,    0.961529402270719f,     -0.793576233735450f,
  -0.843916713821003f,    0.314105102728384f,     -0.204535560653294f,
  0.753318789613803f,     0.160678386635821f,     -0.647065919861379f,
  -0.202789866826280f,    0.648108234268198f,     -0.261292621025902f,
  0.156681828732770f,     0.405377351820066f,     0.228465381497500f,
  0.972348516671163f,     0.288346037401522f,     -0.0799068604307178f,
  0.916939290109587f,     -0.279220972402209f,    -0.203447523864279f,
  -0.533640046855273f,    0.543561961674653f,     0.880711097286889f,
  -0.549683064687774f,    0.0130107219236368f,    -0.554838164576024f,
  -0.379442406201385f,    -0.00500104610043062f,  0.409530122826868f,
  -0.580423080726061f,    0.824555731914455f,     -0.254134502966922f,
  0.655609706875230f,     0.629093866184236f,     -0.690033250889974f,
  -0.652346551677826f,    0.169820593515952f,     0.922459552232043f,
  0.351812083539940f,     0.876342426613034f,     -0.513486005850680f,
  -0.626382302780497f,    -0.734690688861027f,    0.245594886018314f,
  -0.875740935105191f,    -0.388580462918006f,    0.0127041754106421f,
  -0.0330962560066819f,   -0.425003146474193f,    0.0281641353527495f,
  0.261441358666622f,     0.949781327102773f,     0.919646340564270f,
  0.504503377003781f,     0.0817071051871894f,    0.319968570729658f,
  0.229065413577318f,     -0.0512608414259468f,   -0.0740848540944785f,
  -0.0974457038582892f,   0.532775710298005f,     -0.492913317622840f,
  0.492871078783642f,     -0.289562388384881f,    0.229149968879593f,
  0.697586903105899f,     0.900855243684925f,     0.969700445892771f,
  -0.618162745501349f,    -0.533241431614228f,    -0.937955908995453f,
  0.886669636523452f,     0.498748076602594f,     0.974106016180519f,
  -0.199411214757595f,    0.725270392729083f,     -0.0279932700005097f,
  -0.889385821767448f,    -0.452211028905500f,    -0.487216271217731f,
  -0.577105004471439f,    0.777405674160298f,     0.390121144627092f,
  -0.595062864225581f,    -0.844712795815575f,    -0.894819796738658f,
  0.0556635002662202f,    0.200767245646242f,     0.481227096067452f,
  -0.0854169009474664f,   0.524532943920022f,     -0.880292014538901f,
  -0.127923833629789f,    -0.929275628802356f,    0.233276357260949f,
  -0.776272194935070f,    0.953325886548014f,     -0.884399921036004f,
  -0.504227548828417f,    -0.546526107689276f,    0.852622421886067f,
  0.947722695551154f,     -0.668635552599119f,    0.768739709906834f,
  0.830755876586102f,     -0.720579994994166f,    0.761613532216491f,
  0.340510345777526f,     0.335046764810816f,     0.490102926886310f,
  -0.568989013749608f,    -0.296018470377601f,    0.979838924243657f,
  0.624231653632879f,     0.553904401851075f,     -0.355359451941014f,
  0.267623165480721f,     0.985914275634075f,     -0.741887849211797f,
  0.560479100333108f,     -0.602590162007993f,    -0.874870765077352f,
  -0.0306218773384892f,   0.963145768131215f,     0.544824028787036f,
  -0.133990816021791f,    0.0679964588712787f,    -0.156401335214901f,
  -0.0802554171832672f,   0.856386218492912f,     0.143013580527942f,
  0.403921859374840f,     -0.179029058044097f,    0.770723540077919f,
  -0.183650969350452f,    -0.340718434629824f,    0.217166124261387f,
  -0.171159949445977f,    0.127493767348173f,     -0.649649349141405f,
  -0.0986978180993434f,   0.301786606637125f,     0.942172200207855f,
  0.0323236270151113f,    -0.579853744301016f,    -0.964413060851558f,
  0.917535782777861f,     0.442144649483292f,     -0.684960854610878f,
  -0.418908715566712f,    0.617844265088789f,     0.897145578082386f,
  0.235463167636481f,     0.0166312355859484f,    0.948331447443040f,
  -0.961085640409103f,    -0.0386086809179784f,   -0.949138997977665f,
  0.738211385880427f,     0.613757309091864f,     -0.606937832993426f,
  0.825253298062192f,     0.932609757667859f,     -0.169023247637751f,
  -0.411237965787391f,    0.550590803600950f,     -0.0561729280137304f,
  -0.559663108323671f,    -0.718592671237337f,    0.885889621415361f,
  -0.364207826334344f,    -0.839614660327507f,    0.265502694339344f,
  0.394329270534417f,     -0.270184577808578f,    -0.865353487778069f,
  -0.528848754655393f,    -0.179961524561963f,    0.571721065613544f,
  -0.774363220756696f,    0.251123315200792f,     -0.217722762975159f,
  0.0901359910328954f,    -0.329445470667965f,    0.366410356722994f,
  -0.777512662632715f,    0.654844363477267f,     -0.882409911562713f,
  -0.613612530795153f,    -0.926517759636550f,    0.111572665207194f,
  0.0729846382226607f,    0.789912813274098,      0.784452109264882f,
  -0.949766989295825f,    0.318378232675431f,     0.732077593075111f,
  0.786829143208386f,     -0.134682559823644f,    0.733164743374965f,
  0.978410877665941f,     0.992008491438409f,     -0.319064303035495f,
  0.958430683602029f,     0.514518212363212f,     0.101876224417090f,
  0.642655735778237f,     -0.170746516901312f,    0.252352078365623f,
  -0.761327278132210f,    0.724119717129199f,     0.889374997869224f,
  -0.785987369200692f,    -0.594670344572584f,    0.805192297495935f,
  -0.990523259595814f,    0.483998949026664f,     0.747350619254878f,
  -0.824845161088780f,    0.543009506581798f,     -0.208778528683094f,
  -0.314149951901368f,    0.943576771177672f,     -0.102633559170861f,
  -0.947663019606703f,    -0.557033071578968f,    0.419150797499848f,
  0.251214274356296f,     0.565717763755325f,     0.126676667925064f,
  -0.0213369479214840f,   0.342212953426240f,     -0.288015274572288f,
  0.121313363277304f,     0.452832374494206f,     0.545420403121955f,
  -0.616024063400938f,    -0.0320352392995826f,   -0.400581850938279f,
  0.0642433474653812f,    -0.673966224453150f,    0.951962939602010f,
  -0.241906012952983f,    0.0322060960995099f,    -0.449185553826233f,
  -0.709575766146540f,    0.0283340814242898f,    0.234237103593580f,
  -0.285526615094797f,    -0.793192668153991f,    -0.437130485497140f,
  -0.956132143306919f,    0.601158367473616f,     0.238689691528783f,
  0.173709925321432f,     0.437983847738997f,     0.397380645202102f,
  0.432093344086237f,     -0.0338869881121104f,   -0.966303269542493f,
  0.875351570183604f,     -0.0584554089652962f,   0.294207497692552f,
  0.200323088145182f,     0.826642387259759f,     0.284806025494260f,
  -0.00660991970522007f,  0.682493772727303f,     -0.151980775670668f,
  0.0470705546940635f,    -0.236378427235531f,    -0.844780853112247f,
  0.134166207564174f,     -0.586842667384924f,    0.0711866699414370f,
  0.311698821368897f,     -0.361229767252053f,    0.750924311039976f,
  0.0764323989785694f,    0.898463708247144f,     0.398232179543916f,
  -0.515644913011399f,    -0.189067061520362f,    -0.567430593060929f,
  -0.641924269747436f,    -0.0960378699625619f,   -0.792054031692334f,
  0.803891878854351f,     -0.233518627249889f,    -0.892523453249154f,
  0.707550017996875f,     -0.782288435525895f,    -0.156166443894764f,
  -0.543737876329167f,    0.565637809380957f,     -0.757689989749326f,
  -0.612543942167974f,    -0.766327195073259f,    0.587626843767440f,
  -0.280769385897397f,    -0.457487372245825f,    0.0862799426622438f,
  -0.616867284053547f,    0.121778903484808f,     -0.451988651573766f,
  -0.618146087265495f,    -0.285868777534354f,    0.108999472244014f,
  -0.620755183347358f,    -0.268563184810196f,    -0.721678169615489f,
  -0.146060198874409f,    -0.661506858070617f,    0.901707853998409f,
  0.222488776533930f,     0.679599685031679f,     0.974760448601209f,
  0.535485953830496f,     -0.562345697123585f,    0.369219363331071f,
  -0.0282801684694869f,   -0.0734880727832297f,   0.733216287314358f,
  -0.514352095765627f,    -0.850813063545195f,    0.642458447327163f,
  0.118661521915783f,     -0.907015526838341f,    0.789277766886329f,
  -0.719864125961721f,    0.274329068829509f,     0.830124687647056f,
  0.719352367261587f,     -0.821767317737384f,    -0.840153496829227f,
  0.650796781936517f,     0.381065387870166f,     0.341870564586224f,
  -0.00174423702285131f,  -0.216348832349188f,    0.678010477635713f,
  -0.748695103596683f,    -0.819659325555269f,    0.620922373008647f,
  0.471659250504894f,     0.417848292160984f,     -0.990577315445198f,
  -0.509842007818877f,    0.705761459091187f,     0.723072116074702f,
  -0.606476484252158f,    -0.871593699865195f,    -0.662059658667501f,
  -0.207267692377271f,    -0.274706370444270f,    0.317047325063391f,
  0.329780707114887f,     -0.966325651181920f,    -0.666131009799803f,
  0.118609206658210f,     0.232960448350140f,     -0.139134616436389f,
  -0.936412642687343f,    -0.554985387484625f,    -0.609914445911143f,
  -0.371023087262482f,    -0.461044793696911f,    0.0277553171809701f,
  -0.241589304772568f,    -0.990769995548029f,    0.114245771600061f,
  -0.924483328431436f,    0.237901450206836f,     -0.615461633242452f,
  0.201497106528945f,     -0.599898812620374f,    0.982389910778332f,
  0.125701388874024f,     -0.892749115498369f,    0.513592673006880f,
  0.229316745749793f,     0.422997355912517f,     0.150920221978738f,
  0.447743452078441f,     0.366767059168664f,     -0.605741985891581f,
  0.274905013892524f,     -0.861378867171578f,    -0.731508622159258f,
  0.171187057183023f,     0.250833501952177f,     -0.609814068526718f,
  -0.639397597618127f,    -0.712497631420166f,    -0.539831932321101f,
  -0.962361328901384f,    0.799060001548069f,     0.618582550608426f,
  -0.603865594092701f,    -0.750840334759883f,    -0.432368099184739f,
  -0.581021252111797f,    0.134711953024238f,     0.331863889421602f,
  -0.172907726656169f,    -0.435358718118896f,    -0.689326993725649f,
  0.415840315809038f,     -0.333576262820904f,    0.279343777676723f,
  -0.0393098862927832f,   0.00852090010085194f,   -0.853705195692250f,
  0.526006696633762f,     -0.478653377052437f,    -0.584840261391485f,
  0.679261003071696f,     0.0367484618219474f,    -0.616340335633997f,
  -0.912843423145420f,    -0.221248732289686f,    -0.477921890680232f,
  -0.127369625511666f,    0.865190146410824f,     0.817916456258544f,
  0.445973590438029f,     -0.621435280140991f,    -0.584264056171687f,
  0.718712277931876f,     -0.337835985469843f,    0.00569064504159345f,
  -0.546546294846311f,    0.101653624648361f,     -0.795498735829364f,
  -0.249043531299132f,    -0.112839395737321f,    -0.350305425122331f,
  -0.910866368205041f,    0.345503177966105f,     -0.549306515692918f,
  0.711774722622726f,     0.283368922297518f,     0.0401988801224620f,
  0.269228967910704f,     0.408165826013612f,     -0.306571373865680f,
  0.937429053394878f,     0.992154362395068f,     0.679431847774404f,
  0.660561953302554f,     0.903254489326130f,     -0.939312119455540f,
  -0.211194531611303f,    0.401554296146757f,     -0.0373187111351370f,
  -0.209143098987164f,    -0.483955198209448f,    -0.860858509666882f,
  0.847006442151417f,     0.287950263267018f,     0.408253171937961f,
  -0.720812569529331f,    0.623305171273525f,     0.543495760078790f,
  -0.364025150023839f,    -0.893335585394842f,    -0.757545415624741f,
  -0.525417020183382f,    -0.985814550271000f,    -0.571551008375522f,
  0.930716377819686f,     -0.272863385293023f,    0.982334910750391f,
  0.297868844366342f,     0.922428080219044f,     0.917194773824871f,
  0.846964493212884f,     0.0641834146212110f,    0.279768184306094f,
  0.591959126556958f,     0.355630573995206f,     0.839818707895839f,
  0.219674727597944f,     -0.174518904670883f,    0.708669864813752f,
  -0.224562931791369f,    0.677232454840133f,     -0.904826802486527f,
  -0.627559033402838f,    0.263680517444611f,     0.121902314059156f,
  -0.704881790282995f,    0.242825089229032f,     -0.309373554231866f,
  -0.479824461459095f,    -0.720536286348018f,    -0.460418173937526f,
  0.774174710513849f,     0.452001499049874f,     -0.316992092650694f,
  0.153064869645527f,     -0.209558599627989f,    0.685508351648252f,
  -0.508615450383790f,    0.598109567185095f,     0.391177475074196f,
  0.964444988755186f,     0.336277292954506f,     -0.0367817159101076f,
  -0.668752640081528f,    0.169621732437504f,     -0.440925495294537f,
  0.352359477392856f,     0.300517139597811f,     0.464188724292127f,
  0.342732840629593f,     -0.772028689116952f,    0.523987886508557f,
  0.920723209445309f,     0.325634245623597f,     0.999728757965472f,
  -0.108202444213629f,    -0.703463061246440f,    -0.764321104361266f,
  0.153478091277821f,     0.400776808520781f,     0.0362608595686520f,
  0.602660289034871f,     -0.00396673312072204f,  0.296881393918662f,
  0.563978362789779f,     0.849699999703012f,     0.699481370949461f,
  -0.517318771826836f,    0.488696839410786f,     -0.863267084031406f,
  0.0353144039838211f,    0.346060763700543f,     0.964270355001567f,
  0.354899825242107f,     0.806313705199543f,     0.675286452110240f,
  0.0873918818789949f,    -0.595319879813140f,    0.768247284622921f,
  0.424433552458434f,     -0.308023836359512f,    0.802163480612923f,
  -0.348151008192881f,    -0.889061130591849f,    -0.593277042719599f,
  -0.669426232128590f,    0.758542808803890f,     0.515943031751579f,
  -0.359688459650311f,    0.568175936707751f,     0.741304023515212f,
  0.260283681057109f,     0.957668849401749f,     -0.665096753421305f,
  0.769229664798946f,     -0.0871019488695104f,   -0.362662093346394f,
  -0.411439775739547f,    0.700347493632751f,     0.593221225653487f,
  0.712841887456470f,     0.413663813878195f,     -0.868002281057698f,
  -0.704419248587642f,    0.497097875881516f,     -0.00234623694758684f,
  0.690202361192435f,     -0.850266199595343f,    0.315244026446767f,
  0.709124123964306f,     0.438047076925768f,     0.798239617424585f,
  0.330853072912708f,     0.581059745965701f,     0.449755612947191f,
  -0.462738032798907f,    0.607731925865227f,     0.0898348455002427f,
  -0.762827831849901f,    0.895598896497952f,     -0.752254194382105f,
  -0.0916186048978613f,   -0.906243795607547f,    0.229263971313788f,
  0.401148319671400f,     -0.699999035942240f,    0.949897297168705f,
  0.442965954562621f,     -0.836602533575693f,    0.960460356865877f,
  -0.588638958628591f,    -0.826876652501322f,    0.358883606332526f,
  0.963216314331105f,     -0.932992215875777f,    -0.790078242370583f,
  0.402080896170037f,     -0.0768348888017805f,   0.728030138891631f,
  -0.259252300581205f,    -0.239039520569651f,    -0.0343187629251645f,
  -0.500656851699075f,    0.213230170834557f,     -0.806162554023268f,
  -0.346741158269134f,    0.593197383681288f,     -0.874207905790597f,
  0.396896283395687f,     -0.899758892162590f,    0.645625478431829f,
  0.724064646901595f,     0.505831437483414f,     -0.592809028527107f,
  0.191058921738261f,     0.329699861086760f,     0.637747614705911f,
  0.228492417185859f,     0.350565075483143f,     0.495645634191973f,
  0.0378873636406241f,    -0.558682871042752f,    -0.0463515226573312f,
  -0.699882077624744f,    -0.727701564368345f,    -0.185876979038391f,
  0.969357227041006f,     -0.0396554302826258f,   0.994123321254905f,
  -0.103700411263859f,    -0.122414150286554f,    -0.952967253268750f,
  -0.310113557790019f,    0.389885130024049f,     0.698109781822753f,
  -0.566884614177461f,    -0.807731715569885f,    0.295459359747620f,
  0.353911434620388f,     -0.0365360121873806f,   -0.433710246410727f,
  -0.112374658849445f,    -0.710362620548396f,    -0.750188568899340f,
  -0.421921409270312f,    -0.0946825033112555f,   -0.207114343395422f,
  -0.712346704375406f,    -0.762905208265238f,    0.668705185793373f,
  0.874811836887758f,     0.734155331047614f,     0.0717699959166771f,
  -0.649642655211151f,    0.710177200600726f,     -0.790892796708330f,
  -0.609051316139055f,    0.158155751049403f,     0.273837117392194f,
  0.101461674737037f,     -0.341978434311156f,    0.358741707078855f,
  0.415990974906378f,     0.760270133083538f,     0.164383469371383f,
  0.559847879549422f,     0.209104118141764f,     -0.265721227148806f,
  0.165407943936551f,     0.611035396421350f,     -0.626230501208063f,
  -0.116714293310250f,    -0.696697517286888f,    0.0545261414014888f,
  0.440139257477096f,     0.202311640602444f,     -0.369311356303593f,
  0.901884565342531f,     0.256026357399272f,     -0.673699547202088f,
  0.108550667621349f,     -0.961092940829968f,    -0.254802826645023f,
  0.553897912330318f,     0.110751075533042f,     -0.587445414043347f,
  0.786722059164009,      0.182042556749386f,     0.398835909383655f,
  0.431324012457533f,     0.587021142157922f,     -0.644072183989352f,
  0.187430090296218f,     0.516741122998546f,     -0.275250933267487f,
  -0.933673413249875f,    -0.767709448823879f,    0.00345814033609182f,
  -0.585884382069128f,    0.463549040471035f,     0.666537178805293f,
  -0.393605927315148f,    -0.260156573858491f,    -0.298799291489529f,
  -0.246398415746503f,    0.970774181159203f,     -0.485708631308223f,
  -0.456571313115555f,    -0.264210030877189f,    -0.442583504871362f,
  0.0585567697312368f,    0.955769232723545f,     0.651822742221258f,
  0.667605490119116f,     -0.178564750516287f,    0.744464599638885f,
  -0.0962758978504710f,   -0.538627454923738f,    -0.844634654117462f,
  0.109151162611125f,     -0.862255427201396f,    -0.478955305843323f,
  0.645965860697344f,     0.345569271369828f,     0.930605011671297f,
  -0.195523686135506f,    0.927940875293964f,     0.0529927450986318f,
  -0.537243314646225f,    0.0815400161801159f,    -0.528889663721737f,
  -0.0803965087898244f,   0.722188412643543f,     -0.115173100460772f,
  0.391581967064874f,     0.609102624309301f,     -0.726409099849031f,
  -0.924071529212721f,    -0.424370859730340f,    -0.932399086010354f,
  -0.679903916227243f,    0.398593637781810f,     -0.395942610474455f,
  0.911017267923838f,     0.830660098751580f,     0.485689056693942f,
  -0.634874052918746f,    0.558342102044082f,     0.153345477027740f,
  -0.418797532948752f,    -0.962412446404476f,    0.499334884055701f,
  0.772755578982235f,     0.486905718186221f,     -0.680415982465391f,
  -0.983589696155766f,    0.0802707365440833f,    -0.108186099494932f,
  0.272227706025405f,     0.651170828846750f,     -0.804713753056757f,
  0.778076504684911f,     0.869094633196957f,     -0.213764089899489f,
  -0.289485853647921f,    -0.248376169846176f,    -0.273907386272412f,
  -0.272735585352615f,    -0.851779302143109f,    0.777935500545070f,
  0.610526499062369f,     -0.825926925832998f,    -0.00122853287679647f,
  -0.250920102747366f,    -0.0333860483260329f,   0.562878228390427f,
  0.685359102189134f,     0.909722844787783f,     -0.686791591637469f,
  0.700018932525581f,     -0.975597558640926f,    0.111273045778084f,
  0.0313167229308478f,    -0.185767397251192f,    -0.587922465203314f,
  -0.569364866257444f,    -0.433710470415537f,    0.744709870476354f,
  0.812284989338678f,     -0.835936210940005f,    0.409175739115410f,
  0.364745324015946f,     -0.526496413512530f,    -0.0949041293633228f,
  -0.710914623019602f,    -0.199035360261516f,    0.249903358254580f,
  0.799197892184193f,     -0.974131439735146f,    -0.962913228246770f,
  -0.0584067290846062f,   0.0678080882131218f,    0.619989552612058f,
  0.600692636138022f,     -0.325324145536173f,    0.00758797937831957f,
  -0.133193608214778f,    -0.312408219890363f,    -0.0752971209304969f,
  0.764751638735404f,     0.207290375565515f,     -0.965680055629168f,
  0.526346325957267f,     0.365879948128040f,     -0.899713698342006f,
  0.300806506284083f,     0.282047017067903f,     -0.464418831301796f,
  -0.793644005532544f,    -0.338781149582286f,    0.627059412508279f,
  -0.624056896643864f,    -0.503708045945915f,    0.339203903916317f,
  -0.920899287490514f,    -0.753331218450680f,    -0.561190137775443f,
  0.216431106588929f,     -0.191601189620092f,    0.613179188721972f,
  0.121381145781083f,     -0.477139741951367f,    0.606347924649253f,
  0.972409497819530f,     0.00419185232258634f,   0.786976564006247f,
  0.716984027373809f,     -0.577296880358192f,    0.336508364109364f,
  -0.837637061538727f,    -0.706860533591818f,    0.655351330285695f,
  -0.799458036429467f,    0.804951754011505f,     0.405471126911303f,
  0.485125485165526f,     0.0354103156047567f,    0.774748847269929f,
  0.396581375869036f,     0.420464745452802f,     -0.544531741676535f,
  -0.779088258182329f,    -0.129534599431145f,    0.228882319223921f,
  0.669504936432777f,     -0.158954596495398f,    -0.171927546721685f,
  0.675107968066445f,     -0.201470217862098f,    -0.185894687796509f,
  0.244174688826684f,     0.310288210346694f,     -0.0674603586289346f,
  0.138103839181198f,     0.269292861340219f,     0.121469813317732f,
  0.168629748875041f,     0.581112629873687f,     0.499508530746584f,
  -0.741772433024897f,    -0.311060071316150f,    -0.263697352439521f,
  0.180487383518774f,     -0.800786488742242f,    -0.949903966987338f,
  0.134975524166410f,     0.213364683193138f,     -0.684441197699222f,
  -0.254697287796379f,    -0.260521050814253f,    -0.757605554637199f,
  -0.134324873215927f,    -0.699675596579060f,    0.136000646890289f,
  0.355272520354523f,     -0.712620797948690f,    -0.729995036352282f,
  0.323100037780915f,     -0.718364487938777f,    0.807709622188084f,
  0.289336059472722f,     -0.558738026311145f,    -0.591811756545896f,
  -0.894952978489225f,    -0.354996929975694f,    -0.142213103464027f,
  -0.651950180085633f,    0.182694586161578f,     -0.193492058009904f,
  -0.540079537334588f,    -0.300927433533712f,    0.119585035086049f,
  0.895807833710939f,     -0.413843745301811f,    -0.209041322713332f,
  0.808094803654324f,     0.0792829008489782f,    0.314806980452270f,
  0.502591873175427f,     -0.0474189387090473f,   -0.407131047818007f,
  0.797117088440354f,     0.902395237588928f,     -0.783327055804990f,
  -0.591709085168646f,    0.628191184754911f,     -0.454649879692076f,
  -0.588819444306752f,    0.250889716959370f,     -0.582306627010669f,
  -0.145495766591841f,    -0.634137346823528f,    0.667934845545285f,
  0.873756470938530f,     0.425969845715827f,     -0.779543910541245f,
  -0.265210219994391f,    0.781530681845886f,     0.304461599274577f,
  0.211087853085430f,     0.281022407596766f,     0.0701313665097776f,
  0.342337400087349f,     -0.0618264046031387f,   0.177147380816613f,
  0.751693209867024f,     -0.279073264607508f,    0.740459307255654f,
  -0.352163564204507f,    -0.726238685136937f,    -0.565819729501492f,
  -0.779416133306559f,    -0.783925450697670f,    0.542612245021260f,
  -0.810510148278478f,    -0.693707081804938f,    0.918862988543900f,
  -0.368717045271828f,    0.0654919784331340f,    -0.438514219239944f,
  -0.923935138084824f,    0.913316783811291f,     -0.961646539574340f,
  0.734888382653474f,     -0.464102713631586f,    -0.896154678819649f,
  0.349856565392685f,     0.646610836502422f,     -0.104378701809970f,
  0.347319102015858f,     -0.263947819351090f,    0.343722186197186f,
  0.825747554146473f,     0.0661865525544676f,    0.918864908112419f,
  -0.999160312662909f,    -0.904953244747070f,    0.246613032125664f,
  0.123426960800262f,     -0.294364203503845f,    -0.150056966431615f,
  0.904179479721301f,     0.517721252782648f,     0.661473373608678f,
  -0.784276966825964f,    -0.984631287069650f,    0.239695484607744f,
  -0.499150590123099f,    0.00153518224500027f,   -0.817955534401114f,
  0.221240310713583f,     0.114442325207070f,     -0.717650856748245f,
  -0.722902799253535f,    -0.962998380704103f,    0.214092155997873f,
  -0.226370691123970f,    0.536806140446569f,     0.111745807092858f,
  0.657580594136395f,     -0.239950592200226f,    0.0981270621736083f,
  -0.173398466414235f,    0.414811672195735f,     0.147604904291238f,
  -0.649219724907210f,    0.907797399703411f,     -0.496097071857490f,
  -0.0958082520749422f,   -0.700618145301611f,    0.238049932994406f,
  0.946616020659334f,     0.143538494511824f,     0.0641076718667677f,
  0.377873848622552f,     -0.413854028741624f,    -0.838831021130186f,
  0.0208044297354626f,    0.476712667979210f,     0.850908020233298f,
  0.0692454021020814f,    0.841788714865238f,     -0.251777041865926f,
  0.512215668857165f,     0.827988589806208f,     -0.399200428030715f,
  0.999219950547600f,     -0.00465542086542259f,  0.279790771964531f,
  -0.683125623289052f,    0.988128867315143f,     0.00697090775456410f,
  -0.409501680375786f,    -0.190812202162744f,    -0.154565639467601f,
  0.305734586628079f,     -0.922825484202279f,    -0.0622811516391562f,
  -0.502492855870975f,    0.358550513844403f,     0.678271216185703f,
  -0.940222215291861f,    0.568581934651071f,     0.953835466578938f,
  -0.476569789986603f,    0.0141049472573418f,    0.722262461730185f,
  -0.128913572076972f,    -0.583948340870808f,    -0.254358587904773f,
  -0.390569309413304f,    -0.0495119137883056f,   -0.486356443935695f,
  -0.112768011009410f,    -0.608763417511343f,    -0.145843822867367f,
  0.767829603838659f,     0.791239479733126f,     0.0208376062066382f,
  -0.524291005204912f,    -0.200159553848628f,    -0.835647719235620f,
  -0.222774380961612f,    0.0617292533934060f,    0.233433696650208f,
  -0.543969878951657f,    -0.628168009147650f,    -0.725602523060817f,
  -0.273904472034828f,    0.320527317085229f,     -0.556961185821848f,
  0.261533413201255f,     0.696617398495973f,     0.200506775746016f,
  -0.395581923906907f,    0.0196423782530712f,    -0.0552577396777472f,
  -0.594078139517501f,    -0.816132673139052f,    -0.898431571909616f,
  0.879105843430143f,     0.949666380003024f,     -0.245578843355682f,
  0.528960018663897f,     0.201586039072583f,     -0.651684706542954f,
  0.596063456431086f,     -0.659712784781056f,    -0.213702651629353f,
  -0.629790163054887f,    -0.0572029778771013f,   0.645291034768991f,
  -0.453266855343461f,    -0.581253959024410f,    0.632682730071992f,
  0.991406037765467f,     0.701390336150136f,     -0.879037255220971f,
  -0.304180069776406f,    0.0969095808767941f,    -0.465682778651712f,
  0.815108046274786f,     -0.532850846043973f,    0.950550134670033f,
  0.296008118726089f,     -0.198390447541329f,    0.159049143352201f,
  0.105169964332594f,     -0.482506131358523f,    0.493753482281684f,
  0.292058685647414f,     0.283730532860951f,     0.00323819209092657f,
  0.765838273699203f,     -0.0753176377562099f,   -0.679824808630431f,
  0.548180003463159f,     -0.996048798712591f,    0.782768288046545f,
  0.396509190560532f,     0.848686742379558f,     0.750049741585178f,
  0.730186188655936f,     -0.0220701180542726f,   -0.487618042364281f,
  -0.403562199756249f,    -0.0693129490117646f,   0.947019452953246f,
  -0.731947366410442f,    0.198175872470120f,     -0.329413252854837f,
  -0.641319161749268f,    0.186271826190842f,     -0.0739989831663739f,
  -0.334046268448298f,    -0.191676071751509f,    -0.632573515889708f,
  -0.999115061914042f,    -0.677989795015932f,    0.289828136337821f,
  0.696081747094487f,     0.965765319961706f,     -0.203649463803737f,
  -0.195384468978610f,    0.746979659338745f,     0.701651229142588f,
  -0.498361303522421f,    0.289120104962302f,     0.270493509228559f,
  -0.132329670835432f,    0.385545665843914f,     -0.265371843427444f,
  0.306155119003788f,     -0.859790373316264f,    -0.0198057838521546f,
  0.233572324299025f,     0.426354273793125f,     -0.510901479579029f,
  -0.600001464460011f,    -0.972316846268671f,    0.531678082677910f,
  -0.0543913028234813f,   -0.860333915321655f,    0.0717414549918496f,
  -0.992726205437930f,    0.587189647572868f,     -0.710120198811545f,
  -0.712433353767817f,    0.000905613890617163f,  0.323638959833707f,
  -0.148002344942812f,    0.422217478090035f,     -0.512122539396176f,
  -0.652032513920892f,    -0.0749826795945674f,   0.689725543651565f,
  0.00708142459169103f,   0.612282380092436f,     -0.182868915402510f,
  -0.478704046524703f,    0.630078231167476f,     -0.201694106935605f,
  0.471080354222778f,     0.705764111397718f,     0.504296612895499f,
  -0.245442802115836f,    -0.0255433216413170f,   0.244606720514349f,
  -0.620852691611321f,    0.130333792055452f,     -0.0404066268217753f,
  0.533698497858480f,     0.569324696850915f,     -0.0208385747745345f,
  -0.344454279574176f,    0.697793551353488f,     0.223740789443046,
  0.0819835387259940f,    -0.378565059057637f,    0.265230570199009f,
  -0.654654047270763f,    -0.959845479426107f,    -0.524706200047066f,
  -0.320773238900823f,    0.133125806793072f,     0.204919719422386f,
  0.781296208272529f,     0.357447172843949f,     -0.295741363322007f,
  -0.531151992759176f,    -0.928760461863525f,    -0.492737999698919f,
  0.185299312597934f,     0.0308933157463984f,    0.0940868629278078f,
  -0.223134180713975f,    -0.728994909644464f,    0.748645378716214f,
  0.602424843862873f,     0.939628558971957f,     -0.601577015562304f,
  -0.178714119359324f,    0.812720866753088f,     -0.864296642561293f,
  0.448439532249340f,     0.423570043689909f,     0.883922405988390f,
  -0.121889129001415f,    -0.0435604321758833f,   -0.823641147317994f,
  -0.775345608718704f,    -0.621149628042832f,    0.532395431283659f,
  -0.803045105215129f,    0.897460124955135f,     0.432615281777583f,
  -0.0245386640589920f,   -0.822321626075771f,    -0.992080038736755f,
  -0.829220327319793f,    0.125841786813822f,     0.277412627470833f,
  0.623989234604340f,     -0.207977347981346f,    -0.666564975567417f,
  0.419758053880881f,     -0.146809205344117f,    0.702495819380827f,
  0.802212477505035f,     0.161529115911938f,     0.987832568197053f,
  -0.885776673164970f,    -0.608518024629661f,    -0.126430573784758f,
  0.168260422890915f,     -0.517060428948049f,    -0.766296586196077f,
  -0.827624510690858f,    -0.149091785188351f,    -0.643782325842734f,
  0.768634567718711f,     0.815278279059715f,     -0.648037361329720f,
  -0.480742843535214f,    0.983809287193308f,     -0.701958358623791f,
  0.0797427982273327f,    0.903943825454071f,     0.980486658260621f,
  0.207436790541324f,     -0.536781321571165f,    -0.885473392956838f,
  -0.626744905152131f,    0.279917970592554f,     -0.489532633799085f,
  0.402084958261836f,     -0.566738134593205f,    -0.0990017532286025f,
  0.458891753618823f,     0.893734110503312f,     0.541822126435773f,
  -0.856210577956263f,    -0.0354679151809227f,   -0.868531503535520f,
  0.150589222911699f,     0.611651396802303f,     0.524911319413221f,
  0.555472734209632f,     -0.723626819813614f,    -0.162106613127807f,
  0.602405197560299f,     0.903198408993777f,     0.329150411562290f,
  -0.806468536757339f,    -0.671787125844359f,    -0.707262852044556f,
  0.474934689940169f,     -0.379636706541612f,    0.404933387269815f,
  0.332303761521238f,     0.394233678536033f,     -0.0366067593524413f,
  0.904405677123363f,     -0.356597686978725f,    -0.623034135107067f,
  0.572040316921149f,     0.799160684670195f,     -0.507817199545094f,
  -0.533380730448667f,    -0.884507921224020f,    -0.00424735629746076f,
  0.647537115339283f,     0.456309536956504f,     -0.766102127867730f,
  -0.625121831714406f,    0.341487890703535f,     -0.360549668352997f,
  -0.229900108098778f,    -0.666760418812903f,    0.813282715359911f,
  0.115522674116703f,     -0.221360306077384f,    0.0297293679340875f,
  0.00682810040637105f,   0.0115235719886584f,    0.887989949086462f,
  0.792212187398941f,     0.415172771519484f,     -0.600202208047434f,
  0.949356119962045f,     -0.526700730890731f,    0.946712583567682f,
  -0.392771116330410f,    0.0144823046999243f,    -0.649518061223406f,
  0.724776068810104f,     0.00920735790862981f,   -0.461670796134611f,
  0.217703889787716f,     0.846151165623083f,     -0.202702970245097f,
  0.314177560430781f,     -0.761102867343919f,    0.0528399640076420f,
  -0.986438508940994f,    -0.595548022863232f,    -0.430067198426456f,
  0.150038004203120f,     0.738795383589380f,     -0.605707072657181f,
  -0.597976219376529f,    0.375792542283657f,     -0.321042914446039f,
  0.902243988712398f,     0.463286578609220f,     -0.739643422607773f,
  0.210980536147575f,     -0.539210294582617f,    0.405318056313257f,
  -0.876865698043818f,    -0.0883135270940518f,   0.0300580586347285f,
  -0.657955040210154f,    0.159261648552234f,     0.288659459148804f,
  0.274488527537659f,     0.646615145281349f,     0.431532024055095f,
  -0.982045186676854f,    -0.777285361097106f,    -0.124875006659614f,
  0.503004910525253f,     0.824987340852061f,     -0.859357943951028f,
  -0.894837450578304f,    0.744772864540654f,     0.415263521487854f,
  0.337833126081168f,     -0.612498979721313f,    0.391475686177086f,
  0.573982630935632f,     -0.391044576636065f,    0.669493459114130f,
  -0.763807443372196f,    -0.898924075896803f,    -0.969897663976237f,
  -0.266947396046322f,    0.198506503481333f,     -0.355803387868922f,
  0.787375525807664f,     0.655019979695179f,     -0.266247398074148f,
  -0.665577607941915f,    0.0617617780742654f,    -0.303207459096743f,
  0.807119242186051f,     -0.864431193732911f,    0.711808914065391f,
  0.267969697417500f,     -0.643939259651104f,    -0.723685356192067f,
  0.887757759160107f,     -0.318420101532538f,    -0.984559057628900f,
  -0.123118506428834f,    0.264872379685241f,     0.258477870902406f,
  -0.727462993670953f,    -0.223845786938221f,    0.683462211502638f,
  -0.989811504909606f,    0.292644294487220f,     -0.926087081411227f,
  -0.801377664261936f,    -0.337757621052903f,    0.356167431525877f,
  0.974619379699180f,     0.456124311183874f,     0.664192098344107f,
  -0.910993234571633f,    -0.484097468631090f,    -0.128534589958108f,
  -0.770879324529314f,    0.320053414246682f,     0.249818479771296f,
  0.0153345543766990f,    0.696352481669035f,     -0.397719804512483f,
  0.670333638147646f,     -0.678192291329959f,    0.190143924397131f,
  0.342035884244954f,     -0.350791038317704f,    0.0218450632953668f,
  0.437133719806156f,     0.659960895652910f,     0.903378806323159f,
  0.855089775229062f,     0.946706092624795f,     0.335540975081955f,
  0.838337968455111f,     -0.102693592034237f,    -0.702102376052106f,
  0.409624309223486f,     -0.654483499569910f,    0.886576641430416f,
  -0.200573725141884f,    -0.461284656727627f,    0.262661770963709f,
  0.867505406245483f,     -0.0688648080253220f,   -0.707487995489326f,
  -0.248871627068848f,    -0.197869870234198f,    -0.243745607075197f,
  -0.244106806741608f,    0.489848112299788f,     -0.0909708869175492f,
  -0.377678949786167f,    0.0385783576998284f,    -0.470361956031595f,
  0.802403491439449f,     -0.408319987166305f,    0.345170991986463f,
  -0.433455880962420f,    0.00950587287655291f,   -0.441888155900165f,
  -0.817874450719479f,    0.818308133775667f,     0.931915253798354f,
  0.818494801634479f,     0.787941704188320f,     0.882012210451449f,
  0.0749985961399193f,    0.259772455233352f,     -0.655786948552735f,
  0.0824323575519799f,    0.980211564632039f,     -0.798619050684746f,
  0.496019643929772f,     -0.727312997781404f,    -0.999839600489443f,
  0.625938920414345f,     -0.561059012154101f,    0.215650518505246f,
  0.121571798563274f,     0.161863493108371f,     -0.340322748036792f,
  0.521792371708641f,     0.655248389359818f,     -0.180967013066484f,
  0.936797969156762f,     0.523749660366580f,     0.764744126333943f,
  0.384701560810431f,     -0.744092118301334f,    0.719721922905938f,
  0.365931545158250f,     -0.720202871171563f,    0.121662048491076f,
  -0.355501954289222f,    0.379420491208481f,     -0.593818415223405f,
  -0.433690576121147f,    -0.766763563509045f,    -0.377445104709670f,
  -0.955638620720410f,    0.309622585052195f,     -0.613767678153186f,
  0.0177719922394908f,    0.362917537485277f,     -0.297613292472489f,
  0.0275561832152067f,    -0.962345352767599f,    0.452866577068408f,
  -0.307485159523065f,    0.931778412136845f,     0.639592220588070f,
  0.00782144244951311f,   -0.0466407334447796f,   -0.134392603781566f,
  0.895314655361308f,     -0.537785271016286f,    0.663926391064792f,
  -0.886126633268266f,    -0.0975129470189278f,   -0.429791706025144f,
  -0.440337831994928f,    -0.00156267573188829f,  0.933113069253665f,
  -0.560704402651437f,    -0.201658150324907f,    0.465819560354530f,
  0.0701448781871696f,    0.859597672251104f,     -0.525851890358272f,
  -0.992674038068357f,    -0.0846761339576128f,   -0.401686794568758f,
  -0.886069686075370f,    -0.480254412625133f,    0.432758053902000f,
  0.168651590377605f,     -0.453397134906684f,    0.340250287733381f,
  -0.972972507965963f,    0.0560333167197302f,    -0.180812774382952f,
  -0.689848943619717f,    -0.427945332505659f,    0.771841237806370f,
  0.329334772795521f,     0.573083591606505f,     0.280711890938316f,
  -0.265074342340277f,    -0.166538165045598f,    -0.0612128221482104f,
  0.458392746490372f,     0.199475931235870f,     0.681819191997175f,
  0.356837960840067f,     0.756968760265553f,     0.763288512531608f,
  -0.890082643508294f,    -0.322752448111365f,    0.799445915816577f,
  -0.956403501496524f,    0.723055987751969f,     0.943900989848643f,
  -0.217092255585658f,    -0.426893469064855f,    0.834596828462842f,
  0.723793256883097f,     0.781491391875921f,     0.928040296363564f,
  -0.468095417622644f,    0.758584798435784f,     0.589732897992602f,
  0.929077658343636f,     0.829643041135239f,     0.0947252609994522f,
  0.554884923338572f,     -0.513740258764285f,    -0.221798194292427f,
  0.499855133319165f,     -0.0237986912033636f,   0.559618648100625f,
  -0.509812142428963f,    -0.444047241791607f,    0.678274420898738f,
  -0.983706185790147f,    -0.295400077545522f,    -0.688769625375228f,
  0.729259863393412f,     0.889478872450658f,     0.928277502215167f,
  -0.429388564745762f,    -0.790568684428380f,    0.930220908227667f,
  -0.796970618648576f,    -0.980240003047008f,    0.0372716153521411f,
  -0.290828043433527f,    -0.303854123029680f,    0.160774056645456f,
  -0.712081432630280f,    0.390787025293754f,     0.981202442873064f,
  -0.679439021090013f,    0.183053153027806f,     0.665002789261745f,
  -0.708708782620398f,    0.254574948166343f,     0.0575397183305137f,
  -0.723713533137924f,    -0.732816726186887f,    0.501983534740534f,
  0.879998734527489f,     0.825871571001792f,     0.920880943816000f,
  0.311565022703289f,     -0.788226302840017f,    -0.223197800016568f,
  0.850662847422425f,     -0.365181128095578f,    0.958907951854379f,
  -0.0421327708909884f,   -0.153860389403659f,    -0.219620959578892f,
  -0.469076971423126f,    -0.523348925540362f,    -0.287762354299832f,
  -0.913332930679763f,    0.403264134926789f,     0.725849051303960f,
  0.743650157693605f,     -0.382580349065687f,    -0.297138545454038f,
  -0.480092092629432f,    0.0412697614821378f,    -0.396203822475830f,
  -0.0721078217568973f,   0.979038611510460f,     -0.766187876085830f,
  -0.344262922592081f,    0.943351952071948f,     -0.219460259008486f,
  0.115393587800227f,     -0.342675526066015f,    0.926460460401492f,
  -0.486133445041596f,    0.0722019534490863f,    -0.571069005453629f,
  -0.0854568609959852f,   0.370182934471805f,     -0.554007448618166f,
  0.899885956615126f,     -0.188476209845590f,    -0.548132066932086f,
  0.0559544259937872f,    -0.161750926638529f,    -0.532342080900202f,
  0.585205009957713f,     -0.374876171959848f,    -0.169253952741901f,
  -0.473665572804341f,    0.942267543457416f,     -0.515867520277168f,
  -0.706362509002908f,    -0.320672724679343f,    -0.398410016134417f,
  0.733774712982205f,     0.449599271169282f,     0.109119420842892f,
  -0.285090495549516f,    0.0854116107702212f,    0.0603189331827261f,
  -0.943780826189008f,    0.0679186452322331f,    0.0975973769951632f,
  -0.870728474197789f,    -0.153122881744074f,    -0.519939625069588f,
  -0.633620207951748f,    -0.767551214057718f,    -0.905802311420298f,
  -0.841350087901049f,    -0.271805404203346f,    0.282221543099561f,
  -0.0874121080198842f,   0.0634591013505281f,    0.318965595714934f,
  -0.865047622711268f,    -0.401960840475322f,    0.637557181177199f,
  -0.664578054110050f,    -0.871253510227744,     -0.893972634695541f,
  0.442396058421524f,     -0.427901040556135f,    -0.740186385510743f,
  0.788155411447006f,     -0.541278113339818f,    0.509586521956676f,
  -0.461159620800394f,    0.664671981848839f,     0.880365181842209f,
  -0.0831685214800200f,   0.952827020902887f,     0.183226454466898f,
  -0.176729350626920f,    0.851946105206441f,     -0.361976142339276f,
  0.357209010683668f,     0.982462882042961f,     -0.690757734204635f,
  0.178681657923363f,     -0.0804395784672956f,   0.971787623805611f,
  0.875217157810758f,     0.160844021450331f,     -0.359951755747351f,
  0.0178495935461525f,    0.0203610854761294f,    0.413933338290502f,
  -0.676038601090005f,    -0.111093077131977f,    -0.381792206260952f,
  -0.459903351782575f,    0.308522841938619f,     0.324961267942541f,
  0.365201262605939f,     0.732543185546895f,     -0.559558093250200f,
  0.848266528378337f,     -0.185546299813159f,    0.997052205707190f,
  -0.932554828383249f,    -0.106322273904826f,    -0.0690562674587807f,
  0.919489002936141f,     0.137210930163322f,     -0.664517238270193f,
  -0.985856844408119f,    -0.0719443995256963f,   -0.602400574547167f,
  -0.398979518518077f,    -0.581117055144305f,    -0.0626081333075188f,
  -0.0372763806643306f,   -0.688808592854889f,    0.703980953746103f,
  -0.480647539644480f,    0.615510592326288f,     -0.940226159289884f,
  -0.953483236094818f,    -0.300312284206625f,    -0.819419230573751f,
  0.657560634657022f,     -0.0500336389233971f,   0.628589817614501f,
  0.717012803783103f,     -0.0315450822394920f,   -0.445526173532186f,
  0.521475917548504f,     -0.479539088650145f,    0.695075897089419f,
  -0.0365115706205694f,   0.0256264409967832f,    -0.0121306374106025f,
  -0.817618774100623f,    0.375407640753000f,     0.944299492219378f,
  -0.717119961760812f,    -0.120740746804286f,    0.995225399986245f,
  -0.460846026818625f,    0.904552069467540f,     0.807270804870602f,
  -0.842962924665094f,    -0.923108139392625f,    -0.130295037856512f,
  0.760624035683226f,     0.986419847047289f,     -0.959218334866074f,
  -0.203345611185410f,    -0.474420035241129f,    -0.872329912560413f,
  0.485994152094788f,     -0.515456811755484f,    -0.948541161235413f,
  0.509659433909651f,     0.783030335970347f,     -4.41004028146619e-05f,
  -0.664795573083349f,    0.917509788523214f,     -0.824045573084530f,
  -0.461857767121051f,    -0.667434409929092f,    -0.00822974230444418f,
  0.825606347148302f,     -0.396378080991589f,    0.0161379983198293f,
  -0.940751675506308f,    -0.520997013834332f,    -0.239727035024153f,
  -0.354546027474561f,    0.430652211989940f,     -0.557416557692462f,
  -0.357117274957257f,    -0.891975448321656f,    -0.0775302131779423f,
  0.716775563686830f,     -0.903453980341467f,    0.946455001410598f,
  -0.615060907661003f,    0.964288374469340f,     0.0506144897273089f,
  0.720601612869967f,     -0.991323837622476f,    0.647403093538608f,
  -0.400304988135589f,    -0.883732066109751f,    -0.792060477777513f,
  0.710867542231890f,     -0.840766000234525f,    0.460362174479788f,
  -0.834771343071341f,    -0.329399142231491f,    -0.139853203297018f,
  -0.760035442359396f,    -0.546795911275364f,    -0.598172518777125f,
  0.244198671304740f,     0.0816980976432087f,    -0.978470883754859f,
  -0.425173722072458f,    -0.469868865988971f,    0.847396146045236f,
  0.0513388454446360f,    -0.545662072513986f,    -0.130534232821355f,
  -0.654100097045099f,    0.0409163969999120f,    0.573001152600502f,
  0.706046270983569f,     0.587208280138624f,     0.237670099964068f,
  0.848355476872244f,     -0.318971649676775f,    -0.659343733364940f,
  0.321817022392701f,     -0.595779268050966f,    -0.114109784140171f,
  0.998897482902424f,     -0.615792624357560f,    -0.384232465470235f,
  0.156963634764123f,     0.499645454164798f,     -0.627603624482829f,
  0.169440948996654f,     0.109888994819522f,     -0.492231461622548f,
  -0.463014567947703f,    0.825436145613203f,     -0.0271223123229367f,
  0.497887971992266f,     0.811868354230459f,     -0.192668816770168f,
  0.287930938097264f,     0.0283112173817568f,    0.791359470942568f,
  0.365100854153897f,     -0.566922537281877f,    0.915510517906894f,
  0.674211624006981f,     0.505848146007678f,     0.509348889158374f,
  -0.0477364348461706f,   0.409703628204478f,     -0.820970358007873f,
  -0.565377675052345f,    0.810052924776160f,     -0.448904038826591f,
  -0.830251135876445f,    -0.660589978662428f,    -0.890196028167542f,
  0.130526506200048f,     0.924600157422957f,     0.587215078998604f,
  0.727552064386916f,     -0.224172021948978f,    -0.182984019951690f,
  0.308546229024235f,     0.971188035736775f,     0.0229902398155457f,
  0.0608728749867729f,    -0.0712317776940203f,   0.549832674352445f,
  -0.600015690750697f,    -0.0495103483291919f,   -0.564669458296125f,
  0.726873201108802f,     -0.197851942682556f,    -0.983422510445155f,
  -0.905314463127421f,    0.453289030588920f,     0.792504915504518f,
  -0.840826310621539f,    0.0979339624518987f,    -0.506416975007688f,
  -0.143310751135128f,    -0.451251909709310f,    -0.356156486602212f,
  -0.430777119656356f,    -0.593002001098269f,    -0.212505135257792f,
  -0.378005313269430f,    0.516460778234704f,     -0.574171750919822f,
  -0.702870049350445f,    0.190454765104412f,     0.694962035659523f,
  0.177498499962424f,     -0.00126954773922439f,  -0.766110586126502f,
  -0.769862303237397f,    -0.208905136673906f,    0.0728026097773338f,
  -0.467480087700933f,    -0.368839893652514f,    -0.608806955889496f,
  -0.531329879815774f,    0.411920547737697f,     -0.407318902586407f,
  0.922406353838750f,     -0.0272310683929855f,   0.781051179942937f,
  0.860271807949640f,     -0.703736733439623f,    -0.285650334863399f,
  -0.466904334435873f,    -0.716816768536707f,    0.0869377378786880f,
  -0.280331892461309f,    0.773946156883160f,     -0.139856444064730f,
  0.575680110908147f,     -0.887887626173303f,    0.314286545048942f,
  0.673119170729964f,     0.520399233930039f,     0.581347801663144f,
  0.731708017815653f,     0.672583525027818f,     -0.0534590776637494f,
  -0.880572908687369f,    0.171150522778545f,     -0.377041265530122f,
  -0.478003213002057f,    0.458602883802583f,     0.836824527658741f,
  -0.0686622680764437f,   -0.301000630566919f,    -0.652562984155554f,
  0.604631263268903f,     0.791770979838877f,     0.0790491584346489f,
  0.812646960034949f,     0.138794042671596f,     0.709411730079774f,
  0.226484869016811f,     0.797388098554019f,     -0.162225991160828f,
  -0.0295749256270541f,   0.218242165083417f,     0.442845427695148f,
  -0.480622209857766f,    0.873464432574125f,     -0.868017543466245f,
  -0.435489784247438f,    0.0589001507244313f,    0.829134536020168f,
  0.614063504046069f,     -0.0498036542372153f,   -0.803122689381969f,
  -0.495207870035615f,    -0.126836582496751f,    -0.0715271574335641f,
  -0.600815700055194f,    0.434993547671690f,     -0.891665893518364f,
  0.515259516482513f,     0.475325173737397f,     -0.716548558025405f,
  -0.881097306400870f,    0.738462585443836f,     -0.244486212870867f,
  -0.750368936394211f,    0.303496411011494f,     -0.602701428305057f,
  -0.400346153635480f,    -0.300002744969481f,    -0.518552440201900f,
  0.437964598712580f,     -0.816689813412280f,    -0.814392666138757f,
  -0.888568091915377f,    0.449416911306476f,     -0.231889259488176f,
  0.589775175288682f,     0.817224890217553f,     0.518646001325967f,
  -0.406046689874425f,    -0.822100925750380f,    0.0528571826460145f,
  0.502410576690672f,     -0.795964394123106f,    0.0587614583641718f,
  -0.960750994569408f,    0.0366871534513058f,    0.723018804498087f,
  0.0607565140068052f,    0.337380735516841f,     0.810682513202583f,
  -0.636743814403438f,    0.287171363373943f,     -0.651998050401509f,
  -0.913606366413836f,    0.642186273694795f,     -0.197674788034638f,
  -0.261253290776174f,    0.696450222503413f,     -0.178859131737947f,
  -0.388167582041093f,    -0.0593965887764258f,   -0.638517356081890f,
  0.804955770174156f,     0.220726627737384f,     0.263712659676167f,
  -0.214285245576410f,    -0.267640297291737f,    -0.268009369634837f,
  -0.957726158424482f,    0.708674977585603f,     0.336764494287156f,
  -0.985742981232916f,    -0.883053422617300f,    0.560301189759340f,
  -0.692967747323003f,    0.977419052658484f,     0.0749830817523358f,
  0.916618822945019f,     0.941660769630849f,     0.454145712080114f,
  0.176036352526593f,     0.103229925297037f,     0.936507745325933f,
  -0.870159095287666f,    -0.106465234217744f,    0.684178938709319f,
  0.669775326656340f,     -0.620857222834950f,    0.939074959093680f,
  -0.592224920792423f,    0.620706594809134f,     0.0456831422421473f,
  0.738727999152789f,     -0.751090911501446f,    0.683669216540363f,
  0.153825621938168f,     -0.255671723273688f,    -0.773772764499189f,
  -0.667753952059522f,    0.887641972124558f,     -0.664358118222428f,
  0.512196622998674f,     -0.0234362604874272f,   0.942878420215240f,
  -0.406617487191566f,    -0.140379594627198f,    -0.0587253931185765f,
  0.419570878799757f,     0.533674656399007f,     0.108777047479414f,
  -0.695880604462579f,    0.481525582104998f,     0.511165135231064f,
  0.136105196996658f,     -0.481918536916982f,    0.757546893769363f,
  0.957648176032083f,     -0.908743619686586f,    -0.395640537583668f,
  0.0493439519763970f,    0.293569612893396f,     0.387420368421925f,
  0.0928482742403196f,    0.407302666835821f,     -0.787979245337637f,
  -0.968269218296593f,    -0.409517247978962f,    0.775076200793689f,
  -0.217738166217447f,    -0.370002483875998f,    -0.570975789421316f,
  0.844070553036478f,     0.668620483679341f,     0.00139813137293987f,
  -0.0912495442122028f,   -0.0375370940595317f,   0.723007849224616f,
  0.369999774115317f,     0.862240371150479f,     0.749525689790910f,
  0.742992309993137f,     -0.495813719545874f,    -0.101947508108870f,
  -0.152536889610560f,    0.0598123624723883f,    -0.436496899502871f,
  0.520026918467263f,     0.241005798945400f,     0.970456690492966f,
  -0.376417224463442f,    0.614223236672359f,     0.336733945081746f,
  0.376602027190701f,     0.00373987228923456f,   -0.415425448787442f,
  0.330560415319813f,     -0.277250467297048f,    0.861008806111330f,
  -0.00655914035278493f,  0.810375656135324f,     -0.0113631690466840f,
  -0.191699616402287f,    -0.808952204107388f,    0.813180054552450f,
  0.472985418265257f,     0.180147510998781f,     -0.262580568975063f,
  0.211152909221457f,     -0.882514639604489f,    -0.575589191561861f,
  0.106927561961233f,     0.964591320892138f,     0.738192954342001f,
  0.687649298588472f,     -0.229142519883570f,    -0.354434619656716f,
  -0.420522788056562f,    0.684638470896597f,     -0.608080686160634f,
  0.172668231197353f,     0.571295073068563f,     -0.202258974457565f,
  0.183035733721930f,     -0.425589835248751f,    -0.181955831301366f,
  0.798193178080558f,     -0.719799491928433f,    -0.376418218727565f,
  0.100370714244854f,     -0.674685331738723f,    -0.528950922374114f,
  0.480443520097694f,     0.432497368954013f,     0.887439714903326f,
  0.598241701759478f,     -0.250064970303242f,    -0.743111010477448f,
  0.936189907099845f,     -0.867383557331633f,    0.852536175309851f,
  -0.426378707286007f,    0.793838638663137f,     0.856262917294594f,
  0.734157059815547f,     0.00452009494051664f,   -0.884258713402709f,
  -0.0835595438259760f,   -0.735457210599502f,    -0.710727075357488f,
  0.858050351034768f,     -0.626070522205317f,    -0.848201957131499f,
  0.0180933910837406f,    -0.0350884878366737f,   -0.893836321618480f,
  -0.0682788306189803f,   -0.539993219329871f,    -0.557660404374917f,
  0.268969847256868f,     0.505363999910409f,     -0.0464757944714727f,
  -0.529689906951922,     -0.138445378586710f,    0.992531054118938f,
  0.974585450054910f,     0.940349645687053f,     0.648085319100986f,
  -0.410736404028701f,    0.804131759246012f,     -0.774897101314247f,
  0.178246382655493f,     -0.361699623232501f,    -0.836093509684016f,
  0.806309487627613f,     -0.758182371322663f,    0.718410035716663f,
  -0.213136487421868f,    -0.0563465521625497f,   0.0411192849612654f,
  -0.532497330327019f,    -0.0419839515250475f,   0.769432068229678f,
  0.253556234192255f,     -0.745131216530268f,    -0.890639235422577f,
  -0.140643637034330f,    0.318127074868768f,     -0.497415632768561f,
  -0.383508820416842f,    -0.468783454456628f,    -0.289531078129000f,
  -0.0831555730758713f,   0.0107128404847427f,    -0.567754537918270f,
  0.926366772604370f,     -0.600154724486768f,    -0.0920759547805206f,
  0.889582307602381f,     -0.0710437157605615f,   -0.182724716112986f,
  0.228135065644420f,     0.851015495602628f,     0.653035806598961f,
  -0.986676404958677f,    -0.871714951288816f,    -0.824734086356281f,
  -0.490239304888267f,    0.244318295619814f,     -0.923794688606381f,
  0.670566388343457f,     0.849438492633058f,     -0.225318912425116f,
  0.461075616917687f,     0.656436404012820f,     -0.416403369651597f,
  0.205630417444150f,     -0.163509095777762f,    -0.0670299490212758f,
  -0.315561491397908f,    -0.0952855008191476f,   -0.377993693497066f,
  0.860172853824826f,     -0.669622978211317f,    0.595058880455053f,
  -0.425661849490015f,    -0.0405359106780283f,   0.129968697438974f,
  -0.156244199842099f,    0.996996665434629f,     -0.888570357356090f,
  -0.925646614993414f,    -0.753998082238076f,    0.714491335460749f,
  -0.307849905639463f,    0.536274323586448f,     -0.462944722411129f,
  0.622202376598447f,     -0.215582012734053f,    -0.115115003363232f,
  0.128168110175570f,     -0.556263623663708f,    0.921813264386344f,
  -0.288173574121268f,    -0.175054002159610f,    0.0621862747516269f,
  -0.468862899314091f,    0.976184545317535f,     0.468469061953779f,
  0.679394669665911f,     -0.0651943232114096f,   0.872740953203360f,
  -0.917720162541254f,    0.271535917769933f,     0.265441861283112f,
  0.542190484993772f,     -0.0208550501604048f,   0.983272473294640f,
  -0.522164666401537f,    0.833823680455458f,     0.414337644113416f,
  0.588576354535126f,     0.318369292694380f,     0.870442561030567f,
  -0.422722224743553f,    -0.200185003922166f,    -0.770185495487048f,
  -0.878134057034045f,    -0.712873198675798f,    0.647706512601268f,
  0.593648188899773f,     0.126171748161942f,     -0.189622212946038f,
  0.707877641788638f,     0.790070498218410f,     0.698576567863428f,
  0.594748885238005f,     0.567439045931572f,     -0.591839707769224f,
  -0.632709967090349f,    0.415471238430617f,     0.115403276784208f,
  -0.375797954748234f,    0.123611001678020f,     -0.864109581464288f,
  0.115346512920739f,     -0.515581940111704f,    0.880606114362175f,
  0.356011740142007f,     -0.318112820131587f,    0.765766689783476f,
  -0.226772670084743f,    0.442067390135885f,     0.348547568069751f,
  0.862154389627291f,     -0.894863284060244f,    0.475714942110286f,
  0.552377629980789f,     -0.0838875341374268f,   -0.227654706745770f,
  0.0998522598030438f,    0.870812229993830f,     -0.518250234958224f,
  -0.0635791579471283f,   -0.284101882205902f,    -0.454751668241269f,
  0.720773434493943f,     0.0756117818245317f,    -0.0572317848090118f,
  -0.692584830354208f,    0.776250173796276f,     0.514052484701885f,
  0.00770839936587864f,   0.775668871262837f,     0.933055956393907f,
  0.0501713700022097f,    -0.922194089981246f,    0.266653852930886f,
  -0.408584553416038f,    0.797066793752635f,     -0.785570848747099f,
  0.931403610887599f,     0.660859952465710f,     -0.630963871875185f,
  -0.673000673345695f,    0.518897255252506f,     -0.342041914345720f,
  0.405613809903414f,     -0.373516504843492f,    -0.208292396009356f,
  0.0510871025610438f,    0.396765368381847f,     0.00537609874241829f,
  0.935717099427788f,     -0.564801066383885f,    -0.907523685862547f,
  0.670551481631625f,     -0.457309616171932f,    0.364001526470449f,
  0.140805524345232f,     -0.349945327329409f,    -0.0361532758624807f,
  -0.304268551311720f,    0.618482952755668f,     -0.0120110524971313f,
  0.106364353621731f,     -0.427587198043230f,    0.464249033810121f,
  -0.808297048471569f,    0.675277128303038f,     -0.0663762607268352f,
  -0.431951364170808f,    0.953951718476660f,     -0.725934553905574f,
  -0.685163723789561f,    0.164132617720945f,     0.934798872032034f,
  -0.695343627424553f,    -0.420317401094920f,    -0.689247558220342f,
  -0.605894279765940f,    -0.693832779320227f,    0.455037128281788f,
  0.968645000038447f,     -0.0839147410947130f,   0.603463489419899f,
  0.776913738299999f,     -0.491560292499776f,    0.692235227850848f,
  0.0824017593921889f,    0.459024952691847f,     -0.918050509352710f,
  -0.777463066447746f,    -0.161045596440278f,    0.982603547894360f,
  0.700884888820475f,     0.998304481713913f,     -0.362488733430088f,
  0.171493948866881f,     0.565871153533442f,     -0.965620428705067f,
  -0.835532968802398f,    0.885598629033760f,     0.609604257914327f,
  0.725300244775050f,     0.153524048564152f,     -0.662541112390878f,
  0.912145212201290f,     0.135610445338421f,     -0.0813934125800109f,
  0.242209063597546f,     -0.264886126609115f,    -0.335070345839122f,
  0.823958964903978f,     -0.313110855907701f,    -0.354068037633970f,
  -0.0381190024996405f,   0.117794735211134f,     -0.604442743379238f,
  0.524930955656444f,     -0.754959642694882f,    -0.359151666678207f,
  -0.247910739722172f,    0.573570999369016f,     0.543167570010806f,
  -0.718553346110069f,    0.202415372555816f,     -0.860091438569300f,
  -0.0125446132328610f,   0.509348782140749f,     0.349261188228469f,
  0.424395913611831f,     0.0557092265870811f,    0.740276822496471f,
  0.479158001215769f,     -0.221873518706244f,    -0.744883456979009f,
  0.393114117430743f,     -0.733203119089531f,    -0.506531269498885f,
  -0.505532097672033f,    -0.509440981371663f,    0.666118722468113f,
  0.0164067375520756f,    -0.530276655546078f,    0.786338654343788f,
  -0.985008085364936f,    0.479988836226036f,     -0.233652481382475f,
  0.838641098910395f,     -0.407379719374768f,    -0.314266358910263f,
  -0.938033692224531f,    -0.627320971378707f,    -0.229174127295511f,
  0.642505983671691f,     -0.387855473250297f,    0.360324209821339f,
  -0.900766206699468f,    0.176676285751262f,     0.833894117554548f,
  -0.0207873177403817f,   -0.202625183820044f,    0.706644325019314f,
  -0.817922707040537f,    -0.242742059004419f,    0.282109349674866f,
  0.0164603911954744f,    -0.504625902855950f,    0.0415496120997125f,
  -0.787777778295785f,    0.362588721999523f,     -0.371357162843751f,
  -0.818375262182416f,    0.727779997467707f,     -0.836502839702384f,
  0.0423869176265037f,    -0.283934686546853f,    0.665864224978728f,
  -0.0428162304637920f,   0.243534621880753f,     -0.803789304599586f,
  0.570866852088607f,     0.340615579467880f,     -0.323456502239327f,
  0.403242371952148f,     -0.0679158901587793f,   -0.866985651416456f,
  -0.439873628406335f,    -0.246357367033863f,    0.436234859832243f,
  0.560714706225535f,     -0.632564381913014f,    -0.316451076258298f,
  -0.977122780282003f,    0.0741405862954117f,    -0.217862250253606f,
  0.887093089232476f,     -0.418281865182365f,    -0.638553415535034f,
  -0.262631979211197f,    -0.567499176465252f,    0.676178859605923f,
  0.933551699581608f,     -0.0139735129263516f,   -0.610719575803582f,
  0.565123751720690f,     0.230672823422021f,     0.323935439339366f,
  0.635142215896104f,     0.981184609133698f,     0.883668802319366f,
  -0.281281673616891f,    0.583204242495555f,     0.150854689098149f,
  -0.775890223139644f,    0.419951701513177f,     -0.565767744791652f,
  -0.855232478054420f,    0.472188579901153f,     -0.501463211798228f,
  0.727960518524943f,     0.977187851385321f,     0.908113737694915f,
  -0.570200931535418f,    0.716036980035073f,     0.147838037485588f,
  0.218820342222622f,     -0.0673193461152677f,   0.433612519652386f,
  0.449601736390411f,     0.556458722303960f,     0.417345590820787f,
  -0.783345413347895f,    0.858903187230710f,     0.178354025272247f,
  -0.130619018471658f,    0.858282827806003f,     0.508916167873459f,
  0.139535936201634f,     0.240400109521332f,     -0.102942705407161f,
  0.841682417072375f,     -0.696350979494975f,    -0.793644449061670f,
  -0.698273636720141f,    -0.228676865074326f,    -0.195917865828574f,
  -0.306483109792438f,    -0.865320326812636f,    0.659185969805107f,
  -0.368847387975239f,    0.337343722359231f,     0.0723822170210744f,
  0.907475280998826f,     0.515168301614856f,     0.0790167120323961f,
  -0.756697420780699f,    0.966477562469936f,     -0.663190129982788f,
  0.145761851826854f,     0.376079225193173f,     0.631883071958707f,
  -0.956568110802436f,    -0.735990864315730f,    -0.795999578321461f,
  0.958459465243432f,     0.319180429028702f,     -0.907664654881857f,
  0.992381284978014f,     -0.511208110440365f,    -0.714797966909523f,
  -0.717021870210999f,    0.545775837604423f,     -0.0443828768329362f,
  0.333311879948434f,     0.617237628406207f,     -0.0895192882305207f,
  0.506491005527430f,     -0.354205929841282f,    0.777993157224477f,
  -0.667532693120319f,    -0.105006112097613f,    -0.337191911902220f,
  -0.337964429160738f,    0.609014812897482f,     -0.368922911475613f,
  0.889184378947484f,     -0.676392242654630f,    0.429716870038086f,
  0.916751115281822f,     -0.655611878274175f,    0.538928395264007f,
  0.382674384886170f,     0.0580742902089004f,    -0.0124611362991478f,
  -0.0240388340005702f,   -0.726296501832402f,    -0.805701334732693f,
  0.945344279474230f,     -0.668066000378724f,    0.761436128738929f,
  -0.314275650172792f,    -0.394331510439346f,    0.262887592668013f,
  0.155064800148016f,     -0.561829218656134f,    -0.491542446753775f,
  0.922248338472926f,     0.574575887413700f,     0.631722295929094f,
  -0.368854197209698f,    0.984780657580794f,     0.845286034922662f,
  -0.965631634115590f,    -0.435710392440405f,    -0.616488688868478f,
  0.885865616625930f,     0.425733070487506f,     0.776721663555227f,
  -0.0652930284860209f,   -0.734431875923792f,    0.725517937762654f,
  -0.474146253075108f,    0.895357779508529f,     -0.0725048758018345f,
  -0.360185856190223f,    0.559350280666427f,     0.363695103660096f,
  0.152231254765544f,     0.698196273442671f,     0.0518001104801953f,
  -0.139037096279713f,    0.340637636595997f,     0.584243998596814f,
  -0.442304329829130f,    -0.501574294329747f,    0.250155103662225f,
  0.320493999001502f,     -0.150217982700108f,    -0.0381390799255577f,
  0.734760815545772f,     -0.574574233376749f,    0.593440338163725f,
  0.408049858247104f,     -0.0845023181203484f,   -0.855507806920297f,
  -0.473198309372409f,    0.331033392104072f,     0.196445364460658f,
  -0.799745050834061f,    -0.973517526224363f,    0.333748727500822f,
  -0.772356831553232f,    -0.430793038424357f,    0.649852557262489f,
  0.504357958431509f,     0.779588082810134f,     0.0111847677569461f,
  -0.995526851285634f,    -0.676007517368195f,    0.216774012875664f,
  -0.618928775636485f,    -0.418043155155598f,    -0.532063904545563f,
  -0.566979013994587f,    0.246319907266774f,     0.868651379198082f,
  -0.0433430896891542f,   0.463004686009427f,     -0.162112430964754f,
  0.285379745117090f,     0.901512987149549f,     -0.706916206313139f,
  0.685678130935725f,     -0.673017501666538f,    0.0616272859909088f,
  0.147532779463338f,     -0.0108539826652629f,   0.960841184863269f,
  -0.950190006701182f,    0.992171414792924f,     0.715577145884581,
  0.975908103138584f,     -0.769014520827696f,    -0.463212420186382f,
  -0.0761813427220397f,   -0.704830850508594f,    -0.220579724380686f,
  0.840893269946637f,     -0.432181989863148f,    -0.956790418498701f,
  0.122344784859397f,     -0.242133592220528f,    0.908514497715246f,
  0.303653521236872f,     0.756500828196849f,     -0.752207807361831f,
  0.367894642791072f,     -0.702474131286247f,    0.189226989057138f,
  0.401804209353236f,     0.608493473010907f,     -0.437378101171900f,
  -0.158801297891213f,    -0.381027984311046f,    -0.949403985394057f,
  0.370189685252539f,     -0.872655295458655f,    -0.337934909993878f,
  -0.0619622888328213f,   0.352094440420005f,     0.128759637109350f,
  0.432413186229881f,     -0.497474226759161f,    0.552933107875735f,
  0.332665936587804f,     -0.559261497212156f,    -0.886188834336549f,
  0.0170548801295034f,    0.192729852728271f,     -0.674432365770129f,
  -0.526014722983374f,    0.425009223123802f,     -0.186164676538888f,
  0.190362042383007f,     -0.0930204201587825f,   0.794188212052413f,
  -0.243549629178106f,    0.118970185744958f,     -0.216230226310237f,
  0.412570247218594f,     0.659556685538155f,     -0.150540425515543f,
  -0.850858266540316f,    -0.843827815842486f,    0.629298164439457f,
  0.944304062363374f,     -0.117764731240517f,    0.558568737697335f,
  0.731745392387362f,     -0.00413812760139165f,  -0.251933493011685f,
  -0.473346352965658f,    0.178783613032362f,     0.547769344759580f,
  -0.414330113592064f,    -0.550251453379012f,    -0.925253680779905f,
  0.623832825809309f,     -0.494251081521428f,    0.0643361026845244f,
  0.727107898350051f,     0.814864886916156f,     0.0177325632172460f,
  0.749324691554934f,     -0.266301024849295f,    0.675202550635588f,
  -0.0748462128620644f,   -0.747853513216831f,    -0.222563643557406f,
  -0.608884446788701f,    -0.0374135649675464f,   0.852579123003940f,
  -0.585927920129879f,    0.604065857569210f,     0.573072924781108f,
  0.816831955879520f,     0.723975232584095f,     0.367887024581694f,
  0.765292601085641f,     0.836490699448589f,     0.623434131440044f,
  0.743568762340577f,     0.140474444458222f,     -0.746327891490507f,
  0.700496422194197f,     0.549693846244016f,     0.729372970291116f,
  0.728185563682229f,     -0.614909046853182f,    -0.756209631211223f,
  -0.530222413502955f,    -0.312453162783936f,    -0.752364704008701f,
  -0.634475515424180f,    -0.133239439768175f,    0.252790153178337f,
  0.760626105409900f,     -0.838262213452153f,    -0.266093046689486f,
  0.549339088324875f,     -0.278178592347115f,    0.190458706141960f,
  0.906814275056971f,     -0.579827980376046f,    -0.134191470195968f,
  0.244720998349483f,     0.795502128014338f,     0.287019836683889f,
  -0.906277889518234f,    -0.817071933038363f,    0.613378274793081f,
  0.518208081766432f,     -0.388902790616382f,    -0.785778461147273f,
  0.574976429920521f,     -0.283168065839246f,    -0.857322381041868f,
  0.424932015236353f,     0.919756642423073f,     0.412896759578072f,
  -0.976511020897041f,    0.157825653359643f,     -0.0606591903280758f,
  0.508438714729350f,     -0.513115001652116f,    0.881391940997543f,
  -0.129708782033534f,    0.382462819411800f,     -0.538751535594113f,
  0.816770663497783f,     0.869013288394013f,     -0.728381721932439f,
  -0.956736333819522f,    -0.839107107637575f,    0.394821058517234f,
  0.721983518815999f,     -0.0847231453556103f,   0.0206545030491683f,
  0.414707730497861f,     0.246591855656934f,     -0.546187573590839f,
  -0.578957978692654f,    0.162844799084821f,     0.493731081270802f,
  -0.765815587549615f,    0.151613093585910f,     -0.112883397239635f,
  0.879319928900002f,     0.295375250614654f,     -0.505370201033860f,
  -0.635319167339584f,    -0.309818465920078f,    0.768627024018538f,
  -0.544374452091825f,    0.758974060573473f,     -0.106050973670013f,
  0.508616501970226f,     -0.207224226211215f,    0.616842248601645f,
  0.688381226662374f,     0.643728558619948f,     -0.906982649598668f,
  0.526262112978799f,     -0.666644270400075f,    0.314806313630502f,
  -0.292000096972562f,    -0.358353880616007f,    0.156344541906829f,
  0.637606941586786f,     -0.199572501073669f,    -0.669369278061572f,
  0.237513395315133f,     -0.576741807179552f,    0.0750117203638310f,
  -0.633877533594996f,    0.829285089669934f,     0.622345234313277f,
  -0.892617583855908f,    -0.280449892200797f,    0.241147361581176f,
  -0.0784016295955696f,   0.414945819313502f,     0.287238318044040f,
  -0.691458270387106f,    0.597656137422422f,     0.549022082569726f,
  -0.590776624040652f,    0.666740423918019f,     -0.743115212424850f,
  0.164036350785269f,     -0.229427480781113f,    0.283602991107853f,
  -0.533993445778340f,    0.185806116700093f,     -0.317953364055307f,
  0.140412503708198f,     0.280706883979940f,     0.0439806827213221f,
  0.176471515460512f,     -0.614144204292693f,    0.314194083857125f,
  0.519572839644130f,     -0.850547081260782f,    -0.515460990713008f,
  0.353087995032390f,     -0.0241014119925820f,   0.269453276643829f,
  -0.608515317887958f,    -0.777818225534647f,    -0.834277444316067f,
  -0.842707175235771f,    -0.929547602540323f,    -0.884691870945475f,
  0.710591809809692f,     0.143423776629673f,     0.797136471728128f,
  0.233311155245426f,     -0.923169961754164f,    0.627911916101674f,
  -0.338187201367212f,    0.211044396110784f,     -0.443699655795038f,
  0.256593551969761f,     -0.406688684034041f,    0.364900889856600f,
  0.900530571350288f,     -0.160476177153537f,    0.0634217008071056f,
  0.709241599309354f,     -0.789562037599596f,    0.00891621061029158f,
  0.801674768895422f,     -0.704378031949125f,    0.430576706378041f,
  0.796937507044124f,     -0.193348850174576f,    -0.493924902919358f,
  -0.935781577118986f,    0.468142331108629f,     0.00965840085728753f,
  0.0834398764999438f,    0.599712941235232f,     -0.735675950275295f,
  0.200152501800787f,     -0.751603779675650f,    0.0697488403240092f,
  0.300634243862625f,     -0.901969784333300f,    -0.958816237033024f,
  -0.754976119377363f,    0.719702182489622f,     -0.338038642556184f,
  -0.703280944186943f,    -0.579148694005994f,    0.556115731092296f,
  -0.920710928208685f,    -0.278178108839470f,    -0.793795308512285f,
  0.916547680808212f,     0.419467216101691f,     0.177932177026735f,
  0.682833725334600f,     -0.926849803428705f,    0.179045225389745f,
  -0.209414969718359f,    -0.889551871881532f,    0.961659420127890f,
  -0.250341627298645f,    0.105606170554974f,     -0.547860346689080f,
  0.845704098204057f,     0.886892635683680f,     0.768134466622042f,
  -0.954777823503721f,    -0.718106389777233f,    -0.580779231998609f,
  -0.0241800476518665f,   0.815063484178525f,     -0.351971452344303f,
  0.770369263680192f,     0.520886146470712f,     -0.236456125482696f,
  0.0900173391919312f,    -0.00610611501589697f,  0.0986788038317752f,
  0.277083194173223f,     0.0877085076786761f,    0.695814138412262f,
  0.281021332783082f,     -0.701468161850407f,    -0.785496560046616f,
  -0.805623403379156f,    -0.0524204125046179f,   0.0836418099696601f,
  0.467252832788807f,     0.148967572323544f,     0.314141193557124f,
  -0.722297309069329f,    0.147068590429361f,     -0.868307069306109f,
  0.118712645744921f,     0.737896544941878f,     0.897526485681248f,
  0.842207508585120f,     0.817408479766998f,     0.522315328909182f,
  -0.409136979179218f,    0.580654760034574f,     -0.384701243761730f,
  -0.769398544059918f,    -0.791317178699730f,    0.357020281620118f,
  -0.235410423267782f,    -0.326332500533018f,    -0.416891876268284f,
  -0.863029987000052f,    0.505171215727166f,     -0.728709553380428f,
  0.554546891580919f,     0.737429989077498f,     -0.355088598334119f,
  0.911987317939763f,     0.525846127625130f,     0.851549830104189f,
  -0.772303673276796f,    0.0421942169353806f,    -0.521836640530782f,
  0.995279650924240f,     -0.186831960875832f,    0.421233670121556f,
  -0.0891583750230474f,   0.661100169663965f,     0.393809652414978f,
  0.346165179707090f,     0.384203760628548f,     -0.329281932973211f,
  0.446133401546675f,     -0.748200766224366f,    -0.0275154142375615f,
  0.771701580845288f,     -0.0177829993094090f,   0.406813206251131f,
  0.606021648140155f,     0.218435152341115f,     0.236571855064013f,
  -0.513495776515847f,    0.729086381137554f,     -0.137775825035815f,
  0.0320966747364262f,    -0.313487802206023f,    0.105472520924239f,
  0.423606700821375f,     -0.231301628369264f,    0.465218832919270f,
  0.379671652150568f,     -0.00497780485272159f,  0.509290230688327f,
  0.467240127182068f,     0.353587964503845f,     0.390455232684039f,
  0.721536288927627f,     -0.838922323815237f,    0.827628029266859f,
  0.768844149796201f,     -0.813963144642386f,    -0.797054297232628f,
  -0.933039367361175f,    -0.0723957249866136f,   -0.664824147893300f,
  0.695914840901794f,     -0.206071660300270f,    0.879389414398409f,
  0.181872681691416f,     -0.582831210733033f,    0.624249199449935f,
  0.204959730900228f,     0.354831594370532f,     0.337152636438178f,
  0.596132114241829f,     -0.295619496794481f,    -0.443402055665686f,
  0.743995051028396f,     0.543706744165365f,     0.825846155044062f,
  -0.764982315603181f,    -0.0355223730700034f,   -0.682467026736627f,
  -0.914037445162109f,    -0.222823484413727f,    0.825323277024566f,
  0.0769459194171547f,    0.696453968928934f,     0.760786120466962f,
  -0.525470048583831f,    0.764981036001869f,     0.458525204937000f,
  -0.612703584870878f,    0.626016142683351f,     0.284799326870320f,
  -0.130410894642153f,    -0.730659587111424f,    0.0251896513929686f,
  0.744421417725379f,     0.481278905427271f,     -0.718686189713675f,
  -0.972110566787501f,    -0.178005803066219f,    -0.761536801353512f,
  0.675177569459847f,     -0.613068600254845f,    -0.854757540148688f,
  0.641823580903407f,     0.112536000301536f,     0.201235170163357f,
  -0.332623522893231f,    0.602028236317460f,     0.487529253813741f,
  -0.936537443253385f,    0.932862477850079f,     -0.0977461167435834f,
  -0.485449569929182f,    -0.575807340541437f,    -0.920527242558033f,
  -0.938208754460503f,    0.890054000488493f,     -0.154888511872567f,
  -0.106629818916523f,    0.323343252623500f,     0.105328135407289f,
  -0.837197492121459f,    0.497769113944639f,     -0.234127101891878f,
  0.840922493788059f,     -0.994297350473539f,    0.241966031396186f,
  -0.241143860453769f,    -0.598953146106117f,    0.839112451637864f,
  -0.639567866338402f,    -0.219908091959649f,    -0.778137266578287f,
  -0.201424793310289f,    -0.486105622640452f,    0.874947034932591f,
  -0.131437343585340f,    -0.674427373037920f,    -0.161007203320351f,
  0.215285933912207f,     -0.963047650748652f,    -0.841020847986178f,
  0.259702280444602f,     -0.165325097679823f,    0.572379756389254f,
  -0.435802768396928f,    -0.0776125194906274f,   -0.0293182206559168f,
  -0.847945015803839f,    -0.576891917046364f,    0.728544652294888f,
  0.110676857648527f,     0.760459056611184f,     0.486936926897001f,
  0.680603035572503f,     0.330358411271561f,     0.901153157113818f,
  -0.893323547516767f,    0.268679990552354f,     0.794615743189695f,
  0.221637368947158f,     -0.0207579360252996f,   -0.585634995914835f,
  0.587646126395593f,     -0.317780705107399f,    0.790321547328449f,
  0.251610679655279f,     -0.0386445267248654f,   0.881542790650722f,
  -0.469258891944944f,    -0.900544881246558f,    -0.344978220866601f,
  -0.271404539202745f,    0.863631450621357f,     0.805892242474368f,
  -0.325004362330199f,    -0.649692260224921f,    0.535815472185538f,
  0.427767946389023f,     0.924517987543855f,     0.571059970962007f,
  0.549923246060706f,     -0.639468249016352,     0.307213071097954f,
  -0.885892976847170f,    -0.526002656640427f,    0.733743042788359f,
  0.186919211020217f,     0.322167483598106f,     -0.933484010727969f,
  0.307181642341518f,     -0.391959805653480f,    -0.892298105797306f,
  0.100065710151584f,     -0.932962740784651f,    -0.643536993204857f,
  0.200747180046148f,     0.310831344540979f,     -0.923416823619512f,
  0.440768799148345f,     -0.666930667413366f,    -0.485487251971431f,
  -0.0627811951952384f,   -0.331082293469460f,    0.0335811939608148f,
  -0.653610782697787f,    -0.320586426505716f,    0.559163070852115f,
  -0.497363452770543f,    -0.329886484503569f,    -0.146612217140156f,
  -0.0265272745798242f,   -0.288663397675155f,    -0.996138396801714f,
  0.705746028666908f,     0.634215549629091f,     0.165248482682243f,
  -0.110791752682943f,    -0.0583711657160508f,   0.704663932851230f,
  0.105987046073574f,     -0.674234600022039f,    -0.852792911043127f,
  0.779458558047699f,     -0.506163961277651f,    0.661431789813829f,
  0.362986600662932f,     0.677673397902417f,     0.909704544299484f,
  -0.678129611146149f,    -0.700854916363125f,    -0.954905799366644f,
  0.819329178422143f,     -0.278866438326573f,    0.240572863896085f,
  -0.597973444252616f,    0.520707363092687f,     -0.891796539359942f,
  -0.0707113684027092f,   0.730270237241197f,     -0.202809887987925f,
  0.712903235793333f,     0.815918058519912f,     -0.619284883130692f,
  0.620432327799984f,     0.215462902206797f,     0.913706499476201f,
  -0.284266999538807f,    0.137669223817851f,     -0.320599930994154f,
  -0.279885143029947f,    0.0759863610502050f,    0.362519848337183f,
  0.0897184432777523f,    0.730407126330006f,     -0.715664883515070f,
  -0.964294244830797f,    0.337668374417089f,     0.563780948124681f,
  0.534272089774928f,     0.670003495251274f,     0.976582736706313f,
  -0.576021162432801f,    0.318863740329612f,     0.374838616807691f,
  0.437628782275460f,     0.629331465907672f,     0.800673318445353f,
  -0.964950925853698f,    -0.115288102568929f,    0.581179798077059f,
  0.892103220665649f,     -0.224009831257430f,    -0.486848659265476f,
  0.768601825625188f,     -0.478996958061453f,    0.987216084861456f,
  -0.00828256241998737f,  0.443388113322642f,     -0.209225960405120f,
  0.784392408672073f,     -0.821157008960409f,    0.169088259578466f,
  0.188648958653604f,     0.796321723736402f,     0.804915614204973f,
  -0.947435972579018f,    -0.320368366702004f,    -0.0857043727442930f,
  -0.229914128505395f,    -0.802013870592427f,    0.497444368231634f,
  0.791040716463223f,     0.586369970276563f,     0.871236424247704f,
  0.770091868124107f,     -0.458396647683594f,    0.871149873224889f,
  0.753895449519495f,     0.295832468734546f,     0.574616471536691f,
  0.384408809311353f,     -0.978021020306570f,    0.0397482936794495f,
  0.628095200786834f,     -0.968059492217325f,    -0.404306711220928f,
  0.659301030460980f,     -0.345766174675525f,    -0.0517956907600681f,
  -0.640289082986305f,    0.965202733073502f,     0.909703156044274f,
  -0.744545066259015f,    -0.676836498528477f,    0.0507393165493961f,
  0.394673166210436f,     0.250366706754377f,     -0.287411651947684f,
  -0.521760552601739f,    0.214487178617345f,     -0.922260536787078f,
  -0.970217444063294f,    -0.632705652784150f,    -0.720016326822300f,
  -0.506393579710801f,    0.774172771450182f,     0.891546338793249f,
  0.559706491124446f,     -0.513481979527671f,    0.735727350850420f,
  -0.207760672132971f,    0.956672164225499f,     -0.516696999265124f,
  -0.846015525317730f,    -0.199370940530009f,    0.927580907007946f,
  0.669786891276299f,     -0.208316500739886f,    -0.349932032863852f,
  0.382722440637189f,     -0.455635180187178f,    -0.573852668753046f,
  0.237990995216907f,     -0.00210628303929439f,  0.846035951941252f,
  0.921932267818374f,     0.141873820779935f,     0.871317167610738f,
  -0.632607355185838f,    -0.565801401210940f,    -0.959881482283947f,
  -0.732559764685905f,    -0.655277252471118f,    0.136770193226314f,
  0.206392768880907f,     0.0946932052352707f,    -0.147722827344946f,
  0.142504821799194f,     -0.891443939735724f,    -0.660161817562772f,
  -0.918683225740157f,    0.524851053279394f,     -0.841532325411647f,
  -0.662931925252737f,    0.450018807591706f,     0.157794014139767f,
  -0.562525486647545f,    0.604051451992330f,     0.859220943805127f,
  0.943321402026900f,     0.511188518123118f,     -0.332990520726740f,
  0.904709059147998f,     -0.336911302156504f,    -0.0329301811082998f,
  0.307263624236174f,     -0.640655394434152f,    0.791676792853669f,
  0.450137270831791f,     0.746000232170803f,     -0.915436267533878f,
  0.976514418439799f,     0.828073391112522f,     0.990695018409237f,
  0.419713963781614f,     -0.286897378037841f,    0.111527193084439f,
  -0.956913449095442f,    0.263769440437253f,     0.534739246489713f,
  -0.918314908283506f,    0.680501951418845f,     -0.0258330390798596f,
  -0.696521999550769f,    0.274590593565720f,     -0.821334538131451f,
  0.104139627465949f,     -0.790104923997319f,    0.399265830301725f,
  0.118854169469537f,     0.309552488812324f,     -0.961100729890863f,
  -0.665645274594184f,    -0.125767140532335f,    0.377154316156289f,
  -0.971986633153292f,    -0.148225730575294f,    -0.801072242848910f,
  0.735673216754228f,     0.247753694178141f,     0.759093842520115f,
  -0.529946694334253f,    0.594235069164038f,     -0.801015868726278f,
  0.141962211231124f,     0.135473683510959f,     -0.0431660672944612f,
  -0.437176231417910f,    0.467008031415084f,     0.324675317141816f,
  0.122578305547357f,     -0.0351479470228342f,   -0.437236315511244f,
  -0.822621846670407f,    0.989461679354308f,     -0.242059902390237f,
  0.800837521050356f,     -0.387832478851607f,    0.316362161826139f,
  0.602440060427024f,     0.890992007298149f,     0.319686042477150f,
  0.930326885903916f,     -0.170779817104763f,    -0.437602177375177f,
  0.835764105962134f,     0.522922752459604f,     0.295156847627349f,
  -0.857646277751538f,    -0.451421990712551f,    0.752856133268497f,
  -0.826193868550830f,    -0.906961130052697f,    0.118621494342013f,
  -0.627099634988204f,    0.163256363060383f,     -0.719362770410877f,
  -0.576563943780491f,    -0.369939711177846f,    -0.294180430088591f,
  0.868430622614485f,     0.945955651201780f,     -0.879259966782947f,
  0.376142233233261f,     -0.549019623646418f,    -0.366403875933169f,
  -0.631308623984507f,    -0.398270064613022f,    0.631780765950599f,
  -0.497821177556814f,    -0.0754938097555216f,   0.358298259390762f,
  -0.438971283619577f,    -0.835962846436280f,    0.771544885338102f,
  0.132031497593111f,     0.0964144932127649f,    -0.171144812197942f,
  0.734241841669664f,     0.773828279651661f,     0.591442573315395f,
  0.449840299498767f,     -0.249196666141921f,    0.910274822633449f,
  -0.623687862912847f,    -0.954398427932048f,    0.700975370671540f,
  -0.128268698036002f,    0.723971772247224f,     -0.239872317271662f,
  0.599101633280873f,     0.323504979356466f,     0.726076237951951f,
  0.775013638477775f,     -0.736157118505210f,    0.681129332563739f,
  -0.989456914597076f,    -0.860559243921100f,    -0.652547050354339f,
  0.227533741410917f,     0.263244425371628f,     -0.412800042549063f,
  -0.774547399227093f,    0.959749220773555f,     0.0285018454625012f,
  0.0260964660594436f,    -0.817249773797516f,    -0.275510098931589f,
  -0.957071090655421f,    0.755874233806472f,     0.0601247360044190f,
  0.155148678178749f,     0.744458452388040f,     0.206143083045583f,
  0.405575258734775f,     0.591273066531951f,     -0.286358679634110f,
  0.168522523380964f,     -0.0740663582251186f,   0.991796969736415f,
  0.00304472789286958f,   0.0955103281360055f,    0.595292305224677f,
  -0.633460800851610f,    0.969720344590438f,     -0.788939516987962f,
  -0.690852963213444f,    -0.751849610482179f,    -0.454105756229298f,
  0.527652178438853f,     -0.249156091787771f,    -0.395486634371019f,
  -0.586329259469701f,    0.774216673365643f,     0.000796185912973479f,
  0.753872935709907f,     0.691883261316931f,     -0.599798140130743f,
  0.140718954973018f,     0.400016581571111f,     -0.412934563119652f,
  0.782683275869451f,     -0.837415681080234f,    0.503344297140354f,
  0.443222186121185f,     -0.869067764953740f,    0.891507832007671f,
  -0.258782538717313f,    -0.592111951047753f,    0.542828422857983f,
  -0.959476625230936f,    -0.373353196174649f,    0.558975637763876f,
  0.848608638566440f,     -0.861701716955403f,    -0.937645215971517f,
  0.0456695238513540f,    -0.643462752057364f,    -0.194887894642735f,
  0.576940690214110f,     -0.889414400002951f,    -0.120401270393403f,
  0.581976128201341f,     -0.914549817300516f,    0.619675229253819f,
  -0.446355411157033f,    -0.686510097388917f,    0.199022704414501f,
  0.0679083509214176f,    0.939286059873160f,     0.919854436895475f,
  -0.921420499961796f,    -0.933865152326639f,    -0.173428453947994f,
  0.0481843697148709f,    0.282408667923603f,     0.411093542307595f,
  0.332739798472214f,     -0.539048264159821f,    -0.704491312083244f,
  -0.502163632960363f,    0.955228344617550f,     0.620064399129425f,
  -0.470222569036376f,    0.754614931250763f,     -0.616308595262807f,
  -0.914574682979899f,    0.624066330640082f,     0.836911269770582f,
  0.913639510454430f,     0.653228461676548f,     -0.269928008555249f,
  0.313006679186127f,     0.984676487220296f,     -0.492012769698267f,
  0.956868299674771f,     0.291679581317590f,     0.0391808383867289f,
  0.572884371819903f,     0.0424011452585180f,    0.955486550514640f,
  -0.402317209279260f,    -0.606465037288902f,    0.547296561663929f,
  -0.262634118959448f,    -0.555413611714328f,    -0.328781770154915f,
  0.145794994289916f,     0.141260540582646f,     -0.451655981927315f,
  0.305553535897825f,     0.828724940454557f,     0.263943455052409f,
  -0.609183422737396f,    0.691170220321907f,     -0.372701931956834f,
  0.750237424665146f,     -0.249353280856890f,    0.379870697565802f,
  0.385751069018950f,     -0.515117494253264f,    0.716937491491901f,
  0.343749563024118f,     -0.462962268225808f,    -0.542579750084113f,
  0.865163879545508f,     0.348358741505572f,     -0.309602240547849f,
  -0.0504864877295679f,   -0.822856269672862f,    0.199343960697129f,
  -0.790668167630170f,    -0.0910655952543342f,   -0.0243531696455832f,
  0.832501734319368f,     0.604933598167068f,     0.899053047900036f,
  0.270668041381131f,     0.523691409964688f,     -0.0841567002292820f,
  -0.844392287920523f,    -0.910987838261586f,    -0.470654231510287f,
  -0.103828495683496f,    0.253788695977573f,     -0.103172947809401f,
  -0.339896741661867f,    -0.447251997825083f,    0.217200476817515f,
  -0.474840886373359f,    0.227876267254650f,     -0.851351819181938f,
  -0.902078585170911f,    0.445464427415683f,     -0.842484493463611f,
  -0.141606736723087f,    0.104224619207891f,     -0.554900879859470f,
  0.818556374444811f,     -0.832710463532413f,    -0.284760316465868f,
  0.697962734672817f,     0.235137001970259f,     0.538298155374871f,
  -0.598477541924834f,    -0.833959821954974f,    -0.164556670763502f,
  -0.443902305525605f,    0.484290717235912f,     0.319356252041167f,
  0.0834544406255109f,    -0.839174383593280f,    -0.514784811627172f,
  0.466424623987191f,     0.597641402168886f,     -0.344706573843316f,
  0.346954604803744f,     0.150560726232471f,     -0.963838773301094f,
  -0.210406119881130f,    0.740751216241446f,     -0.519896828058978f,
  0.882277568799242f,     0.982734995306564f,     -0.691486807580351f,
  -0.120653164608028f,    0.263039860106709f,     -0.472131671311566f,
  -0.469155525952548f,    -0.562705921604020f,    -0.737502946123759f,
  0.151863404645485,      -0.367233093688652f,    0.149585386378220f,
  -0.152980596399920f,    0.572826412281344f,     -0.498718037086228f,
  -0.0794332639424211f,   0.659760386972575f,     -0.574814983564964f,
  0.451329484188896f,     0.473066930128670f,     -0.135151886005125f,
  0.379571405476121f,     -0.308712078323501f,    -0.136843563834117f,
  0.395667583713552f,     0.196238140324408f,     0.588147058383512f,
  0.770505301611929f,     -0.865188840370228f,    0.266437694165002f,
  -0.428134513764013f,    0.661967260527446f,     -0.752421375452379f,
  -0.556389852423621f,    0.424944298468302f,     -0.480554454112605f,
  0.916159659428765f,     -0.112147362457396f,    0.363475545209813f,
  0.698805683596358f,     -0.862382341730295f,    -0.489415523853276f,
  0.453056404353730f,     -0.606183761884457f,    -0.00869682692408680f,
  -0.288739722701460f,    0.487988005841341f,     0.566870040344668f,
  0.0894575138005909f,    0.887832293799319f,     -0.0981274237649674f,
  -0.279935090781560f,    0.506891141525948f,     0.952901245338457f,
  0.458002767525373f,     -0.569410776125351f,    0.849518291873527f,
  -0.585020953514368f,    0.676037258640625f,     0.299076264841081f,
  0.911385441491479f,     -0.954959555659035f,    -0.681285607891366f,
  0.631368118385947f,     0.522268523899537f,     0.900701101674748f,
  -0.647701850365577f,    0.567960815808216f,     -0.138958982219446f,
  0.267024801687456f,     -0.975771109955874f,    0.314682157086949f,
  -0.378801381286130f,    0.665990927256163f,     -0.573674360032848f,
  -0.860450785684384f,    0.516581474078532f,     -0.190844183471714f,
  -0.451971355445856f,    -0.808113003973650f,    0.860446168028895f,
  0.377778958059242f,     0.126949039950121f,     -0.892203650250330f,
  0.572503460980517f,     0.975224974978800f,     -0.202312370945465f,
  0.500665599343084f,     -0.0510413720986291f,   0.353231752436633f,
  -0.805555931906752f,    -0.199761377956955f,    -0.829487282239605f,
  0.0282459088867508f,    0.814545057118991f,     0.557652277921578f,
  0.613951716518862f,     -0.678811366342345f,    0.896500288318877f,
  -0.627622562398925f,    0.802545092571611f,     0.211382709497062f,
  -0.979380222642662f,    0.826784411456488f,     -0.670689878657734f,
  0.788878029765924f,     0.137070906151783f,     0.901907287859132f,
  -0.526217367070263f,    -0.545043827128876f,    0.494756249972086f,
  0.236657948774128f,     0.156603327087660f,     0.516397244064118f,
  -0.325837179590292f,    0.460683385171580f,     -0.196022953760504f,
  -0.441996357332195f,    -0.808932369852494f,    0.291980108741838f,
  -0.833583979826152f,    0.365574438479475f,     -0.797139524158001f,
  -0.0649288183732912f,   -0.000696491493834994f, 0.100125393693922f,
  0.598035350719377f,     -0.312548404453564f,    0.0414605409182345f,
  -0.675913083156432f,    0.236245026389435f,     0.550464243484224f,
  0.193366907856750f,     -0.903654015709839f,    -0.00993172527377806f,
  0.0180900754210873f,    0.880678290110106f,     0.166539520562349f,
  -0.984509466189118f,    0.810283124477894f,     -0.925371921448173f,
  0.193528916069728f,     -0.748644561903135f,    0.534508666819454f,
  0.364436869280188f,     -0.386979667637943f,    0.427958998441480f,
  0.362750270039032f,     0.420886957715891f,     0.0300301961707390f,
  -0.655220626875711f,    0.0504522662127427f,    0.472640818703213f,
  -0.417745816013639f,    0.0689992794158720f,    0.461232479061866f,
  -0.483517586427718f,    -0.411463769964024f,    0.622740736364726f,
  0.659687134578680f,     0.243900134982579f,     -0.684356227282321f,
  -0.688699031115733f,    -0.316032121634021f,    -0.644296362948831f,
  -0.236133265458216f,    0.880259454885881f,     -0.956880609581177f,
  0.737775899964131f,     -0.529059297472703f,    0.794119601436042f,
  -0.375698158660466f,    0.493447663117292f,     -0.752511119115434f,
  -0.941143365329844f,    0.610101048864035f,     0.253791011658991f,
  -0.369994602049336f,    -0.697364270085742f,    -0.681360550250048f,
  -0.571943442128960f,    -0.749697324128684f,    0.611997629275096f,
  0.892727938106141f,     -0.440225399106758f,    0.00196047981855352f,
  0.951252158369648f,     0.0351885308766962f,    -0.471806546113710f,
  -0.657231535594911f,    -0.0873481442406481f,   -0.0341288006282565f,
  0.579184398564974f,     -0.224334624306026f,    -0.298557652719061f,
  -0.509401519638379f,    0.188853505083675f,     -0.321619146497229f,
  -0.613159956450671f,    0.570042044631281f,     0.699213203307007f,
  0.537439231469861f,     0.529440733283839f,     -0.744527226912905f,
  0.362949055807175f,     0.529758698714545f,     -0.114804719889245f,
  0.991089489396930f,     -0.186716454683287f,    -0.218189173574106f,
  -0.0493780858124198f,   -0.928812411399224f,    -0.101855573638590f,
  0.454268528366586f,     0.617591620012079f,     -0.197519518988231f,
  0.0973277590468935f,    -0.185672509894105f,    0.649922648337967f,
  -0.896862900376972f,    0.594999589349510f,     -0.746978997769556f,
  0.590642952628647f,     0.935109901616311f,     -0.293310684054096f,
  0.783281817912060f,     -0.189898897214222f,    0.414859016240278f,
  -0.0858574647662298f,   0.0810260863380805f,    -0.633024441577653f,
  0.248442861097829f,     0.984586601784679f,     0.982811638387854f,
  0.547456083836220f,     0.476239638753291f,     -0.897709902882279f,
  -0.208045489357872f,    -0.860637131636973f,    -0.496740558564284f,
  -0.944185351410090f,    0.157610983944341f,     0.975214099838643f,
  0.550265718083095f,     -0.630360326400067f,    0.672420341653334f,
  -0.897139264107564f,    -0.670556423663785f,    0.298764071000339f,
  -0.310465384759529f,    -0.978153640586955f,    0.189785151994709f,
  0.929291975296760f,     0.758271912876084f,     0.806829662560108f,
  -0.472787451147715f,    -0.802032434276146f,    0.455809631085663f,
  0.985520713417984f,     0.739637167649794f,     0.311705106454777f,
  -0.120539152808323f,    0.977785717545631f,     -0.848554870988208f,
  -0.281179241544089f,    0.931102239520177f,     -0.255243432382956f,
  -0.284952242030900f,    -0.189341152192864f,    0.647573166562597f,
  -0.474203015584843f,    -0.545915610099538f,    0.672696420688916f,
  -0.239274489717776f,    0.956544960216021f,     -0.0858024073600807f,
  -0.758223415922611f,    -0.00817763648068581f,  -0.500893489164054f,
  -0.669386983409311f,    -0.344450617815217f,    -0.728051392792875f,
  0.804121117816188f,     0.00718436691280910f,   0.195237363230272f,
  -0.472485206728796f,    0.642070241911164f,     -0.272384993247314f,
  -0.731715323915071f,    -0.791266589031733f,    0.0339783427570857f,
  0.0696513783219659f,    -0.894169486972683f,    0.00234016305501483f,
  -0.0403382685361653f,   -0.943600572111266f,    -0.788181603936192f,
  0.851406365407377f,     -0.100015982664501f,    0.145502229793638f,
  -0.528736628076536f,    -0.0313760382570432f,   -0.662221611141088f,
  -0.885722031379862f,    -0.744257140212482f,    0.524976313116033f,
  0.186092035304635f,     0.181669793648209f,     -0.606482674165339f,
  0.849303544554227f,     0.226118051135263f,     -0.690025550727719f,
  -0.256543384397548f,    -0.207714017766381f,    -0.447913202664626f,
  0.375270273897879f,     -0.884312586292038f,    -0.0838720085819762f,
  0.969898436757285f,     -0.736808033249456f,    0.668875150485586f,
  -0.599937439969920f,    0.470077288925414f,     0.903135367105719f,
  -0.895619185450694f,    -0.637694108244489f,    0.572669535020987f,
  -0.696211470281632f,    -0.820577518545193f,    0.937364674938455f,
  0.422458818039761f,     -0.593964370461091f,    -0.586264791612426f,
  0.0282373486927521f,    0.298051147134121f,     0.592825359583763f,
  0.716195674857467f,     -0.684008410968338f,    -0.167523841045924f,
  -0.370794208549223f,    0.768054740581884f,     0.997835641681024f,
  -0.366262133888883f,    -0.523114034556271f,    -0.457946740456489f,
  -0.530941146838744f,    0.298744841822404f,     0.390761228562591f,
  0.0871171594445448f,    0.764002674223649f,     0.233966808661423f,
  -0.116573523634048f,    0.426118986433559f,     -0.255934695328716f,
  0.302314199650152f,     -0.254971729124577f,    -0.330865677738578f,
  -0.0840307537517577f,   -0.711910586170446f,    0.622585361690409f,
  0.367595248366733f,     0.422102667722561f,     0.269580206097961f,
  0.707083822001774f,     0.625367208198523f,     -0.729594790471199f,
  0.708679674727951f,     0.00355767003560614f,   0.379158300246371f,
  -0.688791438249760f,    0.261637457245975f,     0.704008781391790f,
  -0.917586017594177f,    0.886443038824615f,     -0.923559496787343f,
  0.360365726214756f,     0.547058288460181f,     -0.279853192856989f,
  -0.996331953899586f,    -0.323735921605962f,    -0.618788277975037f,
  0.314597206161166f,     0.106380963133907f,     -0.235044228453968f,
  0.0406899091091886f,    0.687339428801573f,     0.344837805924860f,
  0.123214914005620f,     -0.735264225932133f,    0.0396243248944774f,
  0.270602083588730f,     -0.316104623194235f,    0.201800731173529f,
  -0.348987679395254f,    0.994312100135549f,     -0.986073454140000f,
  -0.787571177818193f,    0.508460947811657f,     -0.443663972776222f,
  0.800303477136838f,     0.712158443474503f,     0.958364684407633f,
  -0.0512343510942759f,   -0.391095518504938f,    -0.291911155637644f,
  0.721770656984705f,     -0.163541232110535f,    0.0366644501980513f,
  0.700853097239887f,     -0.508089885354834f,    -0.375072588159867f,
  0.161585369564288f,     0.686325557438797f,     -0.113188612544717f,
  0.859354598908873f,     -0.723198679696606f,    0.398879124170303f,
  0.139357627051752f,     0.484780500073663f,     -0.0437501438537016f,
  -0.868783676783105f,    -0.147865612288567f,    -0.116480069295514f,
  -0.986846049950927f,    -0.859405305954576f,    -0.631359938031082f,
  -0.0310065270390489f,   -0.288382201791710f,    -0.500960878568203f,
  -0.805633068309090f,    -0.837604329816134f,    0.0325253228618525f,
  -0.538953832190091f,    0.913844038280417f,     0.681967460199437f,
  -0.656775429658090f,    0.922492558885196f,     -0.689527254640680f,
  0.688263898240070f,     -0.225450858342925f,    0.0287239965989763f,
  -0.407744573364816f,    -0.477326718671529f,    -0.780374037627418f,
  0.500400378743065f,     -0.532646941279704f,    0.999679272201893f,
  0.136003002234441f,     -0.811267727922649f,    -0.585019862511894f,
  0.125465493193590f,     0.203160759437510f,     -0.101322607820275f,
  0.543784310894398f,     0.630139383695983f,     0.775322422120693f,
  0.229262447827729f,     -0.656821799421711f,    0.795940998463793f,
  0.263281283116320f,     -0.377237794697631f,    -0.714267543277316f,
  -0.161924029976839f,    0.804294011825499f,     -0.500488029613262f,
  0.716655543045374f,     -0.709565530287520f,    -0.260746944768714f,
  -0.496886497176178f,    -0.896154699339640f,    -0.891352204187934f,
  0.0589172685048254f,    -0.952496908556348f,    -0.543314015084183f,
  0.0724005345282401f,    -0.132089156895576f,    0.694937364018361f,
  -0.884509342587775f,    -0.944587795707932f,    0.346949362800262f,
  -0.587900264454839f,    0.531217960795664f,     0.404240620498887f,
  0.182769547944683f,     0.804826966991636f,     0.601398794220406f,
  -0.767933817870427f,    -0.329693990599177f,    -0.880648189418561f,
  0.0370834298504716f,    -0.405270662847564f,    -0.551993194163015f,
  0.357335885219159f,     -0.442910616174561f,    -0.978355051725551f,
  -0.638907517841606f,    0.266841057307734f,     0.778698832906031f,
  -0.967180516636130f,    -0.772940622039654f,    -0.268706136695081f,
  -0.326082261974967f,    0.0386785617389067f,    0.576293286973562f,
  0.446884000380730f,     0.396703264915684f,     -0.718633572608705f,
  0.586041202195072f,     -0.791039546767268f,    0.556638124682382,
  0.728711593864679f,     -0.576551104247230f,    0.690227524206044f,
  0.0451432373341216f,    -0.0569690667958747f,   0.877674150343795f,
  -0.268602876493051f,    -0.770720641807978f,    0.630269600593677f,
  0.801702094819180f,     0.177071915997341f,     -0.0764831522886398f,
  -0.476930347674815f,    0.0196833210809626f,    -0.566188434097295f,
  0.309890567123613f,     -0.642682312350471f,    -0.645839718540077f,
  -0.985031719881713f,    0.153028235575708f,     -0.446724738384881f,
  -0.616280949001367f,    -0.306418078463084f,    0.313048512921978f,
  0.944732667717825f,     -0.292311689238647f,    0.263616032352334f,
  0.776777395064071f,     -0.529182830991988f,    -0.418996105801001f,
  0.286960890623362f,     0.588336822287104f,     0.268219370126612f,
  -0.696727535489037f,    0.806089151192541f,     0.0396168299208206f,
  -0.613570658239778f,    0.358002315998429f,     -0.0576147175733950f,
  -0.859664908314368f,    0.930793190364908f,     -0.108955403960031f,
  0.640347446939098f,     0.0301817512477458f,    0.508435547839785f,
  -0.774928250619894f,    0.254548271045827f,     -0.192551571812315f,
  -0.401867317012389f,    -0.136220787532581f,    -0.480363308055205f,
  0.146599399729624f,     0.225767301672040f,     -0.207158678688912f,
  0.763491487133281f,     0.161192803873192f,     -0.574968151683314f,
  -0.454043408746924f,    0.427131132989065f,     0.170648543751820f,
  0.0690597676805780f,    0.0360172652133248f,    -0.244429817416531f,
  -0.973014074152018f,    -0.172642279134011f,    -0.798684796670922f,
  -0.622626145444778f,    -0.743408670602069f,    -0.316057396003030f,
  0.908608689971065f,     0.948356574904685f,     0.573858539226522f,
  0.457065605245418f,     -0.246203048690671f,    -0.750525340546383f,
  0.612971646035183f,     0.951528788403619f,     -0.529776510809815f,
  0.0886901849846271f,    -0.0254136796699882f,   0.978897595553096f,
  0.293893753097695f,     0.620217642132267f,     0.862352989549627f,
  -0.379040515436326f,    0.790157871471479f,     0.147151952442201f,
  0.688271487774812f,     -0.897847532497188f,    -0.0355337105008888f,
  -0.850253422176695f,    -0.0354384862653523f,   -0.625796807949394f,
  0.851730076897135f,     0.294773618291289f,     0.834287219330433f,
  0.0758749738551283f,    0.912613321307355f,     -0.326698079590551f,
  -0.844748577890143f,    -0.685263599922107f,    -0.197029963909655f,
  0.591416614029013f,     -0.130921826828109f,    -0.524292687689084f,
  0.356220524225632f,     -0.150091552835503f,    -0.935232109847821f,
  -0.302103008478127f,    -0.998557516519010f,    -0.477012685701094f,
  -0.882343341754284f,    0.210797034143964f,     -0.963566378978947f,
  -0.855600913755685f,    -0.790231379847513f,    -0.625235937382084f,
  0.106405105589857f,     -0.760544427202586f,    0.0103124858505332f,
  -0.610157345750845f,    0.968354521575116f,     0.602472069136318f,
  -0.216458111191680f,    0.935180184275450f,     -0.369261245032360f,
  -0.289325139062185f,    -0.772389696964545f,    -0.345513639348744f,
  0.135539262008296f,     -0.747409495863324f,    -0.849724942811800f,
  -0.739393030129744f,    -0.0301380087411172f,   0.373808817820448f,
  0.760444548005323f,     -0.365739960428504f,    0.121859476627292f,
  -0.719257541809299f,    -0.136914676340304f,    -0.178479405732130f,
  -0.336676444507223f,    -0.795056125367297f,    -0.0872862684496700f,
  -0.950510559362909f,    -0.395266512078238f,    0.636773305385949f,
  -0.150667208767723f,    0.534401287220298f,     -0.349371424663528f,
  -0.784729313810243f,    -0.0510904599006878f,   -0.938702345462904f,
  0.616929636007953f,     -0.228578318449040f,    0.239101663221907f,
  0.0390879233281141f,    -0.294705782740043f,    -0.847928516841798f,
  -0.0480433695823821f,   0.487351505367245f,     -0.820736333448301f,
  0.128692585024021f,     -0.305133215914817f,    0.344900079505924f,
  -0.764316168982242f,    0.717529584295197f,     0.655848670831377f,
  0.479849611138232f,     -0.107624564628078f,    -0.345816374073252f,
  0.0822414215758816f,    -0.0120870567528208f,   0.475870901669481f,
  -0.00594923432583361f,  0.869227669945672f,     -0.262862047504512f,
  0.272430399676396f,     -0.734262318791166f,    0.980593493214018f,
  0.110413869658192f,     -0.732486564250777f,    0.470756873196238f,
  0.897133387901917f,     -0.151953973158384f,    -0.591296220619271f,
  -0.113167158942796f,    -0.103020520738423f,    0.220384226627647f,
  -0.0570027879342681f,   0.0923157145066511f,    -0.523010309215342f,
  0.385053964060568f,     -0.223938668105458f,    -0.0566497019068211f,
  0.636390081595965f,     -0.753651530578004f,    -0.765450358896516f,
  0.790370075460245f,     0.622949415286967f,     -0.0947634056426396f,
  0.122381201893998f,     -0.138573523511105f,    -0.544298107235542f,
  0.535416341314523f,     -0.341107295330707f,    0.266262786345860f,
  0.620108481133049f,     0.190424987800150f,     0.978559599202704f,
  -0.925772919482004f,    -0.300038300695816f,    0.963372836978511f,
  -0.501235224357981f,    0.828375446308031f,     -0.595716120481773f,
  -0.889271354193173f,    -0.389843123593065f,    0.659433696092409f,
  -0.633476165557619f,    -0.708607689555741f,    -0.737738480460783f,
  0.985245299432648f,     0.976853985813928f,     -0.863072444190232f,
  -0.785830171723126f,    0.309433061520758f,     0.166813366328975f,
  -0.552916412621405f,    0.0385101740167735f,    0.445866961855263f,
  0.222557362424800f,     0.0710515871571971f,    -0.368563489700928f,
  0.317406114361191f,     0.326902000037272f,     0.868261309598320f,
  -0.897838476369198f,    0.664364291232529f,     -0.373333343843574f,
  -0.599809263387549f,    -0.411236387818613f,    -0.118186587264933f,
  0.544960929851182f,     0.395925813072269f,     0.337332244255533f,
  -0.0195528742963547f,   -0.580383437020279f,    0.0779554182143842f,
  -0.902635825594202f,    -0.821554429188969f,    0.869996816042779f,
  0.646142135585380f,     -0.0824693320525758f,   0.643317857725100f,
  -0.903892480205129f,    -0.457595546004975f,    0.540461917564665f,
  -0.467530238695992f,    0.107497588388074f,     -0.122360487746121f,
  -0.276968072230331f,    -0.436413500733568f,    0.0719555518906898f,
  -0.794937479672675f,    -0.641344733876686f,    -0.934734152781945f,
  -0.0610463967348016f,   -0.302623058375597f,    0.281116298309257f,
  0.557459622053789f,     -0.350054779110337f,    0.681853624031498f,
  -0.0454067482892435f,   -0.897204174835461f,    0.0289327275291300f,
  0.664312739864751f,     -0.368814604980581f,    -0.576946854776660f,
  -0.187886132141311f,    0.424385580259236f,     0.257994303715228f,
  -0.567650112011742f,    -0.0453371545575014f,   -0.362909825264387f,
  0.450095578912812f,     -0.713870209574945f,    -0.956583539581944f,
  -0.969891699048729f,    -0.417755773448598f,    -0.230738535348142f,
  -0.153353095644968f,    0.539368458440622f,     0.591116036659417f,
  0.779095541288385f,     -0.578525766017613f,    -0.587777137316663f,
  -0.301051260910212f,    -0.319655538885669f,    -0.343495369437935f,
  0.908167583226333f,     0.764220052027033f,     0.0536418758245909f,
  -0.0529753241803754f,   0.249066042857931f,     -0.840152142252005f,
  -0.529971459254312f,    -0.449462194610696f,    0.467144819001113f,
  -0.500103828192601f,    -0.758390449663076f,    0.369740436821770f,
  0.189153926151852f,     -0.188283227959439f,    -0.427563759945909f,
  -0.186773725840825f,    -0.00989853573399446f,  -0.783648829817413f,
  -0.626450875837851f,    -0.328015817185970f,    0.760383401930071f,
  -0.00804531008117837f,  -0.982799468341000f,    0.392730506677802f,
  0.117799138097530f,     0.351088974844522f,     -0.259750164530173f,
  0.776495358243216f,     -0.703059519879109f,    -0.362866233240751f,
  -0.421345310205860f,    -0.818968876330675f,    0.936887497269786f,
  0.713300632813635f,     0.916608801523944f,     -0.147818975792564f,
  0.317064988534009f,     0.885779227314381f,     -0.897706599297367f,
  0.685423132064732f,     0.907830438936990f,     0.0636614655685575f,
  -0.423018627861747f,    0.411565657893159f,     0.911060408474647f,
  -0.617833142759668f,    -0.709543522964145f,    -0.817633731247023f,
  -0.252433983274424f,    0.160456393103956f,     -0.160765428576997f,
  -0.622001061437904f,    -0.470257555319641f,    0.790643274059634f,
  -0.648181378655916f,    -0.828694900506363f,    -0.0234091767546987f,
  -0.562865077760768f,    0.369299949506391f,     -0.423850142805423f,
  0.520699811923658f,     -0.877662359466779f,    -0.739844704434180f,
  0.300520939787139f,     0.0655718600121620f,    0.970843358712180f,
  -0.634231195336845f,    0.324880041395596f,     -0.479089635857354f,
  -0.196422753715449f,    0.568762754402869f,     0.699215376070842f,
  0.445741923102597f,     0.679868900756090f,     0.107609859752086f,
  -0.980983474461865f,    -0.788419140653730f,    0.0696289436185713f,
  0.00330944186568516f,   0.392265626672398f,     0.803469542460994f,
  0.131029913648810f,     -0.845408454497170f,    -0.754797811352229f,
  -0.824208086798235f,    0.510072775586974f,     -0.809491727769575f,
  -0.0228491196350333f,   0.920014947791232f,     0.441066319826495f,
  0.969846842038360f,     -0.199024726691046f,    0.886564290041856f,
  0.203997575245743f,     0.481547443573126f,     -0.637742489331117f,
  0.0664642070998316f,    0.109187062068770f,     -0.952676759642045f,
  0.309247049771982f,     0.880534651306060f,     -0.269363005485603f,
  0.280012695899358f,     0.853031642671923f,     -0.216236966392235f,
  0.903180305900435f,     0.837949615815047f,     0.748563816043584f,
  0.266735542018788f,     -0.685176037557414f,    0.505893787666761f,
  0.977721983069541f,     -0.667151469253569f,    -0.451774081267849f,
  -0.385755850727233f,    0.681037251596535f,     0.550130384863457f,
  0.704080312734731f,     0.519624533199220f,     0.789651392050294f,
  0.176325856625025f,     0.684011432098839f,     -0.469125761119035f,
  -0.841814129063957f,    -0.901473334652527f,    -0.117747872709914f,
  -0.608533033968273f,    0.199709646080986f,     -0.349430401438670f,
  -0.435162733168206f,    -0.368150014673779f,    0.699084004342174f,
  -0.446068942643995f,    0.197420740774886f,     0.524893584115327f,
  0.706475758890142f,     0.912020785879679f,     -0.820472223153770f,
  -0.334742316079635f,    -0.851724976994477f,    -0.702164662784812f,
  -0.649654462810552f,    0.411435475616403f,     -0.0438368033650360f,
  0.799231452421757f,     0.713371883779316f,     0.252437083518609f,
  -0.685658163265283f,    0.0734649179831324f,    -0.400549431226783f,
  -0.415602545578540f,    0.233864615718965f,     0.828846528739923f,
  0.606577491175688f,     -0.266016048272811f,    -0.619106744484090f,
  -0.690853262778644f,    -0.503499724631377f,    -0.409761822901473f,
  0.0576293548519007f,    0.551582021066584f,     0.132631452787255f,
  -0.838228405334512f,    -0.107475742619267f,    -0.875306852866273f,
  -0.184700469068763f,    -0.317074087896838f,    -0.580912620700556f,
  0.453916157844897f,     0.690470988649940f,     0.712835197480083f,
  0.314786689622726f,     0.759835688452120f,     -0.671090442836235f,
  -0.408277610289776f,    -0.815988422173708f,    0.227854929660384f,
  -0.0482646895577266f,   0.968141192561708f,     0.373896367655818f,
  0.820435826598941f,     0.817746838197885f,     -0.0970819110331989f,
  0.679170154451559f,     -0.577986561676471f,    -0.0523570914231941f,
  -0.776930151133931f,    -0.560456597170701f,    0.927747720961181f,
  0.0350177837302503f,    0.844938034137843f,     0.00849044473190053f,
  0.325089161670337f,     -0.851825175889265f,    0.835251667623832f,
  -0.266397917890485f,    0.108463887056499f,     -0.817868888235156f,
  0.590399913800720f,     0.699274619715208,      0.200782223352391f,
  -0.936155874445214f,    0.218471971175575f,     -0.890402779861849f,
  0.268496441855317f,     0.881231954583528f,     0.279360358017994f,
  -0.492400368838405f,    -0.894376670076375f,    0.585129064098519f,
  0.340135248071744f,     0.455880107692993f,     -0.861081993524584f,
  -0.303321115935151f,    -0.562781799622214f,    -0.526041750296426f,
  0.999581943964160f,     0.249814139040315f,     -0.0537475603822974f,
  -0.845239239849439f,    -0.874024176808607f,    0.997751771128387f,
  -0.861617607547820f,    0.671357923629889f,     -0.687974310115279f,
  -0.969462039056016f,    -0.448304961870341f,    0.713064428261850f,
  -0.00718668165564318f,  -0.450608596544700f,    -0.106059234376561f,
  -0.591961308554238f,    0.588633089685867f,     -0.755341317752403f,
  -0.542715401462936f,    0.759199260356047f,     0.0297710796506234f,
  -0.997343196630657f,    0.574076752994254f,     -0.696719940193256f,
  -0.852227517176613f,    0.906332566627663f,     -0.171801252847090f,
  -0.925131151948528f,    -0.0212194634560026f,   -0.940316444070044f,
  0.262965279952363f,     0.902198615594563f,     -0.265057066430189f,
  0.161983092277652f,     0.0181345459457500f,    0.467973650469608f,
  0.857351800575040f,     -0.889882538061811f,    0.728868283859490f,
  0.671187732362764f,     -0.296882575397444f,    -0.793099233276668f,
  0.335561922676737f,     0.0671874495572633f,    -0.0857142329385701f,
  -0.352870876674233f,    -0.119927139078065f,    0.814127111105761f,
  -0.323910302649634f,    -0.313495077982818f,    0.0690526899468447f,
  0.877155536890319f,     0.768040884649443f,     0.158910636324140f,
  -0.824414709871474f,    0.00718921022841235f,   -0.868917281154898f,
  -0.564967532196669f,    0.206261416621150f,     -0.0699574404456100f,
  -0.0547095858591442f,   0.811674902353136f,     -0.562993920383635f,
  0.441212008804309f,     0.917951119557396f,     0.915571961092301f,
  0.0901952529553498f,    0.614118141118295f,     0.760473529905706f,
  -0.566505475760865f,    0.00880029006400429f,   0.975626259597421f,
  0.370738159620831f,     -0.0242162976348563f,   0.828887690189252f,
  -0.665240810020082f,    0.00123256686221063f,   0.184020074202841f,
  0.829917510366750f,     -0.447854906466885f,    0.529356328938248f,
  -0.995192699858126f,    -0.843748622724646f,    -0.422765372440245f,
  -0.386179414096638f,    0.206325400140261f,     -0.369817591904938f,
  0.266933785902425f,     0.892617584642659f,     0.740018647415220f,
  -0.481907279471296f,    0.248268418729551f,     -0.382770749117505f,
  0.974424303757207f,     -0.879320252286332f,    -0.0294961755317245f,
  0.638693329623790f,     -0.765127178629299f,    -0.160881380476610f,
  -0.725001019123526f,    0.00294709357263234f,   -0.701949969294570f,
  -0.708933381768328f,    -0.463893635537772f,    0.476650147791524f,
  -0.206043208566879f,    0.223011684523516f,     -0.258637160422673f,
  0.206325908651728f,     -0.432336904344548f,    0.921979975841259f,
  -0.944396630315761f,    -0.00680582426415510f,  0.319263487872783f,
  -0.836389324192867f,    0.111532890274445f,     -0.938142383682239f,
  -0.637288670131655f,    -0.834211558255576f,    0.251969378874330f,
  -0.970874587083192f,    0.831662411079802f,     -0.446568187924869f,
  -0.659109068071113f,    -0.877869176622375f,    -0.890670252448197f,
  0.477602927742628f,     0.324737705007923f,     -0.147513413112549f,
  -0.186594638422632f,    -0.282864808082840f,    0.745093922271927f,
  0.915500859154332f,     0.0421588655873384f,    -0.483320910754088f,
  0.00503734690385604f,   0.555792895688253f,     0.129412601050279f,
  -0.229347983583150f,    -0.680101211823600f,    -0.866063899229274f,
  0.437769924839021f,     0.133958234316391f,     0.589233411145099f,
  -0.498053917701437f,    0.180863681584405f,     0.525955777469479f,
  -0.581250985307273f,    -0.327934857804250f,    0.482381204171926f,
  -0.867703472610278f,    0.833733008515087f,     -0.607761820334944f,
  -0.758512235503178f,    0.0380785706067470f,    0.719862150842292f,
  0.651283470517919f,     -0.614218162858801f,    -0.239754124815405f,
  -0.733992057859951f,    -0.422541764223845f,    0.951215428883086f,
  0.882569470276544f,     0.937054481646402f,     0.184532408731968f,
  -0.104097666585483f,    0.693277433170057f,     0.800241936558839f,
  -0.998230532922071f,    0.259835639125661f,     0.562745639592536f,
  0.220441127510705f,     0.313735993201991f,     0.330940415696351f,
  -0.602872424656300f,    0.841677792852844f,     0.749701489563795f,
  0.266727039860087f,     0.696379094133993f,     -0.430719144952456f,
  -0.276768289732264f,    -0.0872580230244173f,   -0.722033206227688f,
  -0.837309584159114f,    -0.629739366225350f,    -0.185692585028452f,
  -0.110619837317415f,    0.515881116042359f,     -0.105875685978079f,
  -0.513700186568578f,    0.961245417898430f,     0.655513716233953f,
  -0.0921704793645632f,   -0.694925472850399f,    -0.872174817305748f,
  0.0307133806779607f,    0.531120672076921f,     0.965271277398122f,
  -0.00974420246777163f,  -0.497322783064087f,    0.693565685926388f,
  0.546918707342947f,     -0.230039497490898f,    -0.316024461029338f,
  0.684231559582941f,     -0.306362794944468f,    0.861366189035942f,
  0.378922635334764f,     0.259443877770437f,     -0.838617128408830f,
  -0.205350631644011f,    -0.139772960377519f,    -0.192918167939180f,
  0.602404904043886f,     -0.537407583974730f,    -0.877007125624351f,
  0.361539942609439f,     -0.732030207831016f,    -0.488792995226420f,
  0.612591017966442f,     0.567185560938756f,     0.195543595335781f,
  -0.428955670554558f,    -0.666590144318038f,    -0.702467396810860f,
  -0.894350832807439f,    -0.0620405855731709f,   -0.583114546325259f,
  -0.482155957064968f,    0.212152442925647f,     0.112603107288251f,
  0.0683986906619714f,    0.639176340917929f,     0.642610005510521f,
  -0.708605273163374f,    0.739594669131005f,     -0.492786220480274f,
  -0.308196102291547f,    0.918748221553053f,     0.186736140989674f,
  0.438437026242591f,     0.638769573344929f,     0.928896220524135f,
  0.579945520523175f,     0.218608554904045f,     -0.526070140579576f,
  -0.140303420071590f,    0.304347769360423f,     0.488123173638490f,
  0.987207018313181f,     -0.536397951752998f,    -0.553296120219359f,
  0.184294880372153f,     -0.101502970339396f,    0.287041514309517f,
  0.658172721877726f,     -0.270141883431914f,    -0.0196021946303913f,
  0.000779126872975988f,  -0.0500294515684538f,   -0.588505226599557f,
  0.550916571982769f,     0.703271386531766f,     0.982335628009701f,
  0.942133544852489f,     0.690741953320684f,     0.0466423349204477f,
  -0.941178278727504f,    0.121655023640973f,     0.777925151322362f,
  0.132430336075323f,     -0.114812120408198f,    -0.694094073965245f,
  -0.441397675924967f,    -0.187253074701348f,    -0.672248118097589f,
  -0.688869123609503f,    -0.0723581859661586f,   0.553779536791160f,
  0.380610143087564f,     -0.392032089052147f,    -0.709403552653908f,
  -0.607184251637473f,    0.698227587629545f,     -0.272885954851784f,
  0.0736609147840435f,    0.687106303730018f,     -0.230362931709251f,
  0.393640839382244f,     -0.846905732907407f,    0.0727598538725249f,
  -0.0119849190815611f,   0.470122652313157f,     -0.171681529301612f,
  -0.329268850654460f,    -0.433013841687086f,    -0.943499527192280f,
  -0.123404693276305f,    -0.0861435714812342f,   -0.228816973160929f,
  0.0531549757963279f,    0.901446101051298f,     0.470738280922993f,
  0.238383552115632f,     0.292841887198914f,     -0.617423653544601f,
  -0.865786115828523f,    0.586332203179351f,     0.267618252846898f,
  0.888575002575769f,     -0.0220649407038027f,   -0.946385428026066f,
  0.317436113017866f,     -0.277195072909682f,    -0.207326502081016f,
  0.735387675940421f,     0.961386190882120f,     -0.564038045970629f,
  0.840007249305217f,     -0.262593952346269f,    -0.556378761937190f,
  -0.346529850864238f,    0.00895460576800877f,   -0.695431082536551f,
  -0.105261635693881f,    -0.658342101938401f,    -0.631093613961188f,
  0.601639903111316f,     0.886830692209879f,     -0.600591324826329f,
  -0.350296019796741f,    0.294348102011741f,     0.555826495708193f,
  0.216370653207427f,     -0.672654026881445f,    -0.572202359802723f,
  0.202776438466314f,     -0.490708964058038f,    0.0148723360197853f,
  -0.799031226692943f,    -0.221164759306209f,    0.0323674121757880f,
  -0.130290693568615f,    0.613592603765503f,     0.372755498065474f,
  -0.540502917956863f,    -0.740021877141017f,    0.652888612951242f,
  -0.666157898478327f,    0.476156241264794f,     -0.632081251666311f,
  -0.538341981270842f,    -0.275717185193560f,    0.332983363477103f,
  -0.989659450166330f,    0.212868816589688f,     -0.238985653168422f,
  -0.453005976359810f,    -0.805975530848911f,    -0.948192632970312f,
  -0.291329963979224f,    0.549811667826684f,     0.291147979443248f,
  0.909805561757383f,     0.0728533843443158f,    0.737767652888933f,
  0.605331616290165f,     0.274826946403577f,     0.710517586349601f,
  0.666670055891909f,     0.522059053677516f,     -0.553398792071804f,
  -0.406610321679562f,    -0.893232547853708f,    0.549587730399741f,
  0.714498083720551f,     0.281833380830291f,     0.652788061587949f,
  0.825163748516741f,     0.381299333971584f,     -0.485549061474930f,
  -0.881961689917888f,    0.308937809723222f,     -0.524542880617761f,
  0.329114405956449f,     0.434631551667457f,     -0.894732322264538f,
  -0.831528385961058f,    0.669760583803638f,     -0.674650675537928f,
  -0.373119878846435f,    0.456602566684508f,     0.387804792569985f,
  -0.556983911869482f,    0.000826745899317194f,  0.687973801099889f,
  0.0471935422816141f,    0.0768302380434509f,    0.317557055919800f,
  -0.823316513699125f,    0.394699119350099f,     0.609556161256400f,
  -0.0413041171293194f,   -0.244100882405517f,    -0.939678976894569f,
  0.403390183804743f,     -0.933567523933859f,    -0.331149894636631f,
  -0.0265881324103010f,   0.224249195386459f,     0.888271870759308f,
  -0.119845268644579f,    -0.357275416804345f,    -0.597001288429956f,
  -0.486847206619720f,    -0.181232488650601f,    0.115441291842326f,
  -0.599055795186955f,    0.213179364205327f,     -0.205238322081458f,
  -0.373942142629613f,    -0.610680997090469f,    -0.495737765362772f,
  -0.257634306994249f,    0.583708320566486f,     -0.372047136603982f,
  0.953878668619925f,     -0.632595987923462f,    0.452049761997455f,
  0.166602807787896f,     0.773555002555059f,     -0.277154387560832f,
  -0.557129156714301f,    -0.985242402457283f,    -0.441173064787937f,
  0.561221765682284f,     -0.352004972295446f,    0.970292440826449f,
  0.855523836321424f,     -0.528113079339624f,    0.685454746939680f,
  0.322200261898966f,     0.953249967336372f,     0.825673980624808f,
  0.177229970128320f,     -0.728281956776614f,    -0.479030792350269f,
  -0.00697019557862144f,  0.851652517094715f,     0.853865750362844f,
  0.514736989335681f,     -0.943509205199198f,    -0.0524009027225623f,
  -0.0798997671509367f,   -0.355414349557791f,    -0.366273957594958f,
  -0.565729285138989f,    -0.931573923976439f,    0.345119269147864f,
  0.638375370217726f,     0.711524360229150f,     0.331664704859388f,
  -0.986788646426241f,    0.521200596781614f,     0.656290865944842f,
  -0.436907564088290f,    0.305075696150381f,     -0.848337345127939f,
  0.354044695448027f,     0.690691708552038f,     0.900352213238582f,
  0.475181192463882f,     0.219103309687964f,     0.885437995493547f,
  0.421455288320496f,     -0.879874221804522f,    0.893371290952196f,
  -0.545214090169942f,    0.800731783168682f,     0.249421864783476f,
  0.0766192343033301f,    -0.745747520609971f,    -0.613575150364454f,
  -0.700199720327423,     0.0694373671332735f,    0.759953164582251f,
  -0.0973030480378387f,   -0.298615297250225f,    0.0176506580013247f,
  -0.269562553201540f,    -0.405489169051539f,    -0.00491991297033256f,
  -0.0327449030548885f,   -0.688168836745951f,    0.703014457338754f,
  -0.0909491575673764f,   0.738417882180070f,     0.202377973915515f,
  0.338436193625848f,     -0.408790267504483f,    0.611776208408261f,
  -0.711043784659083f,    0.841495665411188f,     -0.0445715899008592f,
  -0.127281559164749f,    -0.778797832908623f,    0.210344625249896f,
  0.287086540530447f,     -0.703702357088620f,    -0.151146112491418f,
  -0.785180444786487f,    0.427963227387140f,     0.873814130606035f,
  -0.344356753075357f,    -0.755726746591465f,    0.846013365191461f,
  0.126678120904524f,     0.166687962199295f,     -0.148273386834835f,
  -0.770559345875477f,    -0.999129219024862f,    -0.223692721084046f,
  -0.652712854614213f,    0.468054498362978f,     -0.911782175948953f,
  0.555084850374905f,     0.103972972463380f,     -0.414021910330282f,
  0.938793897617340f,     0.515461292224815f,     -0.127677414947037f,
  0.510661477088580f,     0.898409443447962f,     0.528096097102698f,
  -0.444620870908750f,    -0.275909952832928f,    -0.516074838791812f,
  0.110104492330694f,     -0.293114842926621f,    -0.596621371059734f,
  0.152807456749103f,     -0.592864305196648f,    0.948295231208874f,
  -0.575278847840010f,    -0.312463646261757f,    0.664597237604897f,
  -0.177619554099550f,    -0.932259652303036f,    -0.295074750863924f,
  0.731539128777660f,     0.860409131570119f,     -0.0947206503071862f,
  0.106073387018718f,     -0.235389180430490f,    -0.494787189603633f,
  -0.536357147973158f,    -0.680862001049455f,    0.618979489665256f,
  0.613893487415732f,     -0.308605775713246f,    0.694789556987429f,
  -0.440049894326668f,    0.908690328690240f,     0.233612239829512f,
  -0.190662564463532f,    -0.344799878911344f,    -0.185877286582818f,
  -0.553543917790750f,    -0.859543533414720f,    -0.996044831818542f,
  0.0388505104043095f,    0.650508591477642f,     -0.425233346101631f,
  -0.576839967180874f,    0.378730359294024f,     0.531713629917424f,
  0.506096660522796f,     0.854779196325727f,     0.725302682547051f,
  -0.414685510902716f,    0.654208477287561f,     0.580368151427426f,
  -0.000356066597174687f, -0.897393734991154f,    -0.845565244312410f,
  0.615044057364182f,     0.0434592638759266f,    0.342119048500289f,
  -0.696414680186901f,    -0.713269554140146f,    -0.580866925323696f,
  -0.290886355957456f,    -0.473082507703548f,    0.517942229000179f,
  -0.846159512055215f,    -0.715410253368047f,    -0.526272663742330f,
  0.114004124940380f,     -0.207397773975621f,    -0.920379649009572f,
  -0.277970833475531f,    -0.636533427057722f,    -0.972531734576472f,
  -0.687000156900366f,    0.872752357637196f,     0.617872391924648f,
  -0.835274231587444f,    -0.383282792481497f,    0.399233665040770f,
  -0.191230601890140f,    0.620222785371960f,     0.106379326744619f,
  0.987222511696630f,     0.219022023664391f,     0.179689082166371f,
  -0.961619514581522f,    0.570178582343486f,     -0.811091514477978f,
  0.924484469376845f,     0.744507591138529f,     0.272936430096096f,
  0.0646316580619510f,    0.314005111302676f,     0.558833629327024f,
  -0.329744916784918f,    -0.544045568909541f,    0.895769679770795f,
  0.798125821580789f,     0.877473384028199f,     0.616163339432501f,
  0.441057381106904f,     -0.642498173762053f,    0.989059595616979f,
  -0.374771110304453f,    0.480877593471524f,     0.904941689893360f,
  0.428742160807762f,     -0.430483645585549f,    0.0830560957640680f,
  0.694220841170708f,     -0.602964792788891f,    -0.522672782287498f,
  0.717494777479591f,     -0.918002255923909f,    -0.454075191574169f,
  -0.378662039464110f,    0.221482629450150f,     0.750918040362614f,
  -0.636211037178780f,    -0.254529141198887f,    -0.944623201010144f,
  -0.720775773991847f,    -0.674641067104323f,    -0.208243950413264f,
  -0.959488786545901f,    -0.619966503980330f,    0.599486634018692f,
  -0.0955439064236721f,   -0.458181000169795f,    0.736914498713083f,
  -0.176789993854223f,    0.676652697410790f,     -0.967275583857650f,
  0.319377813603719f,     -0.427030468653864f,    0.0670640089595258f,
  0.769945699222976f,     0.767923203047440f,     0.985790354694142f,
  -0.207111795449682f,    0.219134401666738f,     0.548513609112215f,
  0.977227384558063f,     -0.198131173309759f,    0.914163808432723f,
  0.178214485462450f,     -0.240590252223318f,    0.356128697574950f,
  0.453093488702627f,     -0.0401152114159198f,   0.818060948361957f,
  -0.880551400213416f,    0.631519794065582f,     0.658832307703964f,
  -0.179752451562622f,    -0.237844011105596f,    0.739834592198990f,
  0.711355594921083f,     0.774856912009109f,     0.321864249971600f,
  0.470574585274056f,     0.261964793641569f,     -0.634481134262705f,
  0.461363065389595f,     0.0879014163867016f,    0.698353456328335f,
  0.0611830044908546f,    0.918599000791453f,     -0.147822590771951f,
  -0.208296009525534f,    0.775436805889909f,     0.0380914463017457f,
  -0.954468558268744f,    -0.620451283908529f,    -0.770251739379244f,
  0.772246778681563f,     0.326462458587915f,     0.417738473564738f,
  0.0942643452092895f,    0.486153909005530f,     -0.720202618855819f,
  0.0172425211828453f,    -0.460430186764708f,    -0.582933725313246f,
  -0.439721219285309f,    -0.694337374508112f,    0.493516461453915f,
  -0.993527345413430f,    -0.562763570629586f,    -0.0644937992008268f,
  0.741476357523546f,     -0.668588797988340f,    0.594184164979780f,
  -0.605220767543645f,    0.110074204567278f,     -0.599398769115359f,
  0.723882026196765f,     0.678747828159456f,     -0.608589528492249f,
  -0.881419419882399f,    -0.139357674240927f,    0.873828011683502f,
  0.314798068434754f,     -0.457017849147976f,    -0.526003289738433f,
  -0.411404919696823f,    -0.792254466556923f,    -0.299635866135236f,
  0.0102316480137963f,    0.161921266554201f,     0.981427028530907f,
  -0.647351555346480f,    -0.183312260273700f,    -0.348651484808239f,
  -0.198142718294920f,    0.589869434168343f,     -0.201926511662287f,
  0.0337896878721506f,    -0.0276515055864679f,   0.236943449722327f,
  -0.473103622922213f,    0.954358213176107f,     -0.536519478008862f,
  -0.603363977756898f,    0.776267386457251f,     0.780662223932714f,
  0.289187291033147f,     -0.439954328280331f,    0.0429585232791456f,
  0.457321950803212f,     0.236810565417317f,     0.167393310927116f,
  0.634521586990289f,     0.154409349572581f,     -0.750588956901316f,
  0.862647670558265f,     0.800182258889404f,     -0.342011510602950f,
  -0.102697321575297f,    -0.797254530582515f,    -0.718599505627591f,
  -0.729105921762328f,    -0.152424255231618f,    -0.702781451563249f,
  -0.0212710413372206f,   0.961258625954530f,     -0.598484979483616f,
  0.188043416567111f,     -0.511990501189325f,    -0.437449883017104f,
  -0.352443017251219f,    0.0991554004559394f,    -0.663282401319921f,
  -0.835139403797870f,    0.587602722898819f,     -0.939771062270554f,
  0.613878515061637f,     -0.523857415147229f,    0.444842501987166f,
  -0.297001528475358f,    -0.914581150341453f,    0.554844832376064f,
  -0.816400014706997f,    0.823726509832068f,     0.704425080572720f,
  -0.819397910034912f,    0.999003444973468f,     -0.968751535943602f,
  0.0311500939174130f,    0.247867291448898f,     0.835560943875924f,
  0.169794916341582f,     -0.302041142019408f,    0.289549413666482f,
  0.672141268085176f,     0.947060095876251f,     0.324754171403184f,
  0.800014020753458f,     -0.785428883146460f,    -0.463092135879982f,
  0.659192831110219f,     0.118301326248760f,     -0.542297334341874f,
  -0.335957421787428f,    0.794808066256455f,     0.625133567458879f,
  0.227917183877260f,     0.533557157748932f,     -0.948877884679630f,
  0.186417887458649f,     0.859592912781013f,     -0.0183320237921572f,
  0.967066787435574f,     -0.141349529637213f,    0.958107445094614f,
  0.264359167622140f,     -0.631325355674829f,    0.684598042547604f,
  -0.527467468151933f,    0.294659298854560f,     -0.439220168509424f,
  0.391038218778621f,     0.0155669207052447f,    -0.681384294454809f,
  0.146739459198561f,     -0.756404876084652f,    0.381192113543008f,
  0.442850940158445f,     0.964002016096921f,     -0.0507253848694798f,
  0.563462880019551f,     0.190980650425415f,     0.482598778123453f,
  -0.273426091300166f,    0.980640722167518f,     0.198298590133615f,
  0.678100193958147f,     0.530416610025615f,     0.196483886579908f,
  -0.00515783872303177f,  0.0273438459465027f,    -0.257248394117661f,
  -0.576964504105195f,    -0.331030677719652f,    0.389178134459083f,
  0.0714066784585938f,    0.915179137858455f,     0.529738860096996f,
  -0.0851681338619263f,   -0.692212896293625f,    0.0786352959300358f,
  -0.122712774017974f,    -0.154641019547052f,    -0.487537192251297f,
  0.0435645872670241f,    0.856938631597551f,     0.351874085305670f,
  0.708100804109985f,     -0.701200509799317f,    0.0804479422214388f,
  -0.0794375302823220f,   0.543751723132725f,     0.346144383452864f,
  -0.680373368944156f,    -0.572281173045994f,    0.237981706511708f,
  0.0671482960376590f,    0.852393956008547f,     -0.301262907769845f,
  0.523762878044853f,     0.0885512158718469f,    0.885168455552951f,
  -0.333351382431635f,    -0.914187358461713f,    0.657220242471575f,
  0.202238670865175f,     -0.660684692864216f,    0.641271628674064f,
  0.795923699912913f,     -0.332641448887164f,    -0.297595219329770f,
  0.427283618553541f,     0.601893958036382f,     0.355248259075043f,
  -0.420766820174961f,    0.355159952778514f,     -0.806733697216087f,
  -0.694403711049608f,    -0.719250654428532f,    0.580487742419744f,
  0.959156165420351f,     -0.941898541689400f,    0.960568821753178f,
  0.119007749103819f,     -0.973468502734443f,    -0.627534816021182f,
  0.331394418445345f,     -0.415230278112412f,    0.225355270950915f,
  -0.216818510922154f,    0.716553646689289f,     0.149097723527982f,
  -0.212491921692561f,    0.681645638056938f,     0.675358683729395f,
  0.0591550775861416f,    -0.221626142364110f,    -0.235878877821190f,
  0.168188057112471f,     -0.709738432254387f,    0.842890391064944f,
  -0.331175752377862f,    0.231375360302226f,     -0.714989093452242f,
  -0.492645353426504f,    0.552424848261518f,     -0.436987392663331f,
  -0.336155191719795f,    0.137666231065822f,     0.739347397348610f,
  0.493222787180627f,     0.283646543313800f,     -0.603522923409923f,
  -0.474181275984451f,    0.249315354427624f,     0.323736714335287f,
  0.933612934150728f,     -0.651555022796413f,    -0.743229221575077f,
  -0.648309364385349f,    0.115117716036212f,     -0.0689988553878600f,
  0.0394979772968704f,    0.732729774997258f,     0.487584669162102f,
  0.808754952095239f,     0.827617962775983f,     0.550826738558347f,
  0.890858298785235f,     0.152998196795770f,     0.401198245071198f,
  0.187173931669199f,     0.576387011979054f,     -0.464903903379260f,
  0.735172244343599f,     -0.0393734341215035f,   -0.501927105416023f,
  -0.852926247859480f,    0.384774001880198f,     0.723957370923565f,
  0.869614310250896f,     0.698124990202440f,     -0.0618370378422302f,
  -0.273879540781302f,    -0.0745005910544518f,   -0.754408143155094f,
  -0.859084370639359f,    -0.709011936778905f,    -0.883595552533659f,
  0.326386065122049f,     0.756686513420982f,     -0.639817612043620f,
  -0.536531544653662f,    -0.596858657734988f,    -0.187117983404806f,
  0.760208405412209f,     0.191383034225783f,     -0.771443976174702f,
  -0.371171018178012f,    0.723338724416329f,     -0.325113980261468f,
  -0.652823731845602f,    -0.902765567501679f,    -0.109945188610355,
  0.863727536109734f,     0.762531987550249f,     0.484671237555863f,
  -0.376731181566557f,    -0.961176245257487f,    0.374503763045540f,
  -0.275274129954644f,    0.947951135663002f,     0.891610575724484f,
  0.233179187366345f,     0.868694446846928f,     -0.201812205484274f,
  -0.676342903796604f,    0.962133604967067f,     0.0941637112283598f,
  -0.0856261317646829f,   0.375061189807232f,     -0.275342940020193f,
  0.0614298144531287f,    -0.183234253182376f,    0.146964792162229f,
  -0.307180215012337f,    -0.139123531176191f,    0.130840221889238f,
  -0.0654726742084248f,   0.988722897887987f,     -0.805684911622576f,
  0.763299463922693f,     0.148136188784880f,     -0.432183160161832f,
  -0.592185939638987f,    -0.593835208842770f,    -0.366135084813261f,
  0.840566739882685f,     0.572052978307971f,     -0.825682529425410f,
  -0.970222226210689f,    -0.554421263584439f,    0.324648156825255f,
  0.0472246837302466f,    0.168098848238140f,     0.00634984653176796f,
  0.850237261066903f,     0.286624344510407f,     0.196043215794080f,
  0.289161416244007f,     0.334801090322515f,     0.871286740072183f,
  -0.754609531300255f,    0.623871003889383f,     0.0843430009639772f,
  -0.736369938040848f,    0.400507674511444f,     0.816325383600297f,
  -0.500667496861800f,    0.453092855162135f,     0.281798170796444f,
  0.631969623501011f,     0.472467114651372f,     0.525988741184527f,
  -0.124862967293674f,    -0.882904489381606f,    -0.501090007558747f,
  0.631622297793485f,     -0.0234210285578584f,   -0.521093811962915f,
  -0.0402368492672573f,   -0.762999364505356f,    0.948716268452360f,
  -0.572740830308272f,    -0.261042904339051f,    -0.506108365537530f,
  0.585933508412429f,     -0.362463094458446f,    -0.885375028242576f,
  -0.835757117571791f,    0.337250829139564f,     0.298618238243588f,
  -0.744903291826588f,    -0.979848674056393f,    -0.488518944548476f,
  -0.000297116577397283f, -0.137863396173336f,    -0.627207234158244f,
  -0.970417810284170f,    -0.601487862773028f,    -0.999527775716382f,
  0.116672274325216f,     -0.786330829714504f,    0.740118245374718f,
  0.856485463622646f,     -0.555144930193560f,    -0.0168912375666686f,
  -0.774544329159697f,    -0.782767315598991f,    -0.600844843420598f,
  0.885816107471180f,     0.577075799078571f,     0.663829997048111f,
  -0.359000184287277f,    -0.390009578642891f,    0.202240602818017f,
  -0.0191477232064394f,   -0.566459499064884f,    0.288883557382261f,
  0.962583478738218f,     0.782123756762393f,     -0.312311582870785f,
  -0.749354208187204f,    0.205679267602357f,     0.804004517387718f,
  -0.733078779233144f,    -0.426195645938973f,    0.686872484317089f,
  -0.398704803137823f,    -0.267786412313359f,    -0.374306263341615f,
  0.632992513422251f,     -0.972217744254910f,    -0.167080739523409f,
  0.608176739669718f,     -0.935550125875275f,    -0.422451600932096f,
  0.499643952974426f,     -0.491034978653149f,    -0.0256130378373849f,
  -0.158669355267388f,    0.360503946885584f,     0.227714934784132f,
  -0.138648043280479f,    -0.0707461296301128f,   0.0638330442765616f,
  -0.168811643868974f,    -0.575670642767690f,    -0.162143785491822f,
  0.528621079903453f,     0.581283330394272f,     0.444430744183000f,
  0.859288341846780f,     -0.170487584890459f,    -0.440175706710406f,
  -0.184806402672108f,    0.676010805169568f,     -0.0117535553470483f,
  -0.231606756742133f,    -0.210042044569361f,    -0.517950708003565f,
  -0.805772781723687f,    0.156938933772370f,     0.892075905739393f,
  0.403874478002384f,     0.572031508558373f,     -0.604145909072008f,
  -0.330076696654475f,    0.0314560087228033f,    0.683787496948704f,
  -0.788582181996934f,    0.835276281386949f,     -0.0644658492206380f,
  0.938270191882745f,     -0.344927907293928f,    -0.976720519493346f,
  0.906264084343827f,     -0.648152742145255f,    -0.776984965421811f,
  -0.299470572593974f,    -0.423690646950321f,    0.749911693814570f,
  -0.701929894551648f,    -0.665191316321370f,    -0.568359320650352f,
  -0.957309362369509f,    0.914088966355983f,     0.770952996203681f,
  0.0924190787439159f,    0.844599990803978f,     -0.613336716591875f,
  -0.683270165308367f,    0.358563204319583f,     0.934597169812267f,
  0.236596595813630f,     -0.895964332479994f,    -0.673302324943916f,
  0.454883302340070f,     -0.473926010524343f,    -0.576000657136217f,
  -0.644850950007290f,    -0.980218836434995f,    0.321620362364719f,
  -0.799924718666919f,    0.0619872524925393f,    -0.609255645268410f,
  0.159243124858648f,     -0.339764623434603f,    0.379865023026277f,
  -0.923132229333074f,    -0.0300494021321296f,   -0.183835365297645f,
  0.122648511393234f,     0.887652015676064f,     -0.616448517838488f,
  -0.920600866006207f,    0.352861591267815f,     -0.930578364778234f,
  -0.378819076263050f,    0.775423778544869f,     0.836977798656885f,
  0.0472244767469148f,    0.484934339557912f,     -0.939155187409193f,
  0.261555270800537f,     0.143595058480400f,     -0.323517719771947f,
  0.483466454684928f,     -0.423163689969697f,    0.356966814701025f,
  -0.843907304366205f,    0.945903563730962f,     -0.495952298317153f,
  0.972277051575873f,     0.153052037173145f,     -0.715894882755676f,
  -0.617028915483254f,    -0.332307224095366f,    -0.171207102890728f,
  0.841771328272651f,     -0.0308707743261867f,   -0.626480028747696f,
  -0.729235538916864f,    -0.743517330301179f,    -0.733868915239511f,
  -0.449192858200231f,    0.362286468575150f,     0.327436676142902f,
  0.609768663831898f,     -0.147499187968100f,    -0.470195300907973f,
  -0.232167856443943f,    0.225074905574485f,     -0.0818541072414634f,
  0.793403933843056f,     0.267628199755028f,     -0.391701371806294f,
  -0.846991992740029f,    -0.776221590294324f,    0.121351482320532f,
  -0.189789365942677f,    -0.894392208695015f,    -0.632864319945356f,
  0.927817761109627f,     -0.732454610273421f,    0.260011686544283f,
  -0.713973491605344f,    0.469764032416604f,     -0.608895265807545f,
  -0.684992974060601f,    -0.745556289276139f,    -0.536308213076133f,
  0.586581187207818f,     0.149804345860779f,     0.401576742698496f,
  -0.719670291046630f,    0.618659855530024f,     -0.256639783379370f,
  -0.862966031725668f,    0.893866512913152f,     0.861800793529066f,
  -0.704895723095590f,    0.154163397540805f,     -0.0775797186536984f,
  -0.252297335448882f,    0.869851864160888f,     0.428747373815147f,
  -0.818372805928921f,    -0.739117647833389f,    -0.697378012429133f,
  0.182997863108567f,     0.689563104159966f,     -0.0506114067037338f,
  -0.705077813920782f,    0.452892458862023f,     -0.365069844049503f,
  -0.889224821648518f,    0.0194889225677406f,    0.847743515500726f,
  -0.0650338075825718f,   -0.108889937983496f,    -0.168485037502421f,
  0.912533003086865f,     0.428132366084106f,     0.692652998111620f,
  0.130599999674344f,     0.411245435867244f,     -0.194909473459497f,
  0.562152151569866f,     0.503795293326445f,     0.801805532943245f,
  0.795718119772331f,     -0.327975015537058f,    0.771389506217327f,
  0.237139782375987f,     -0.793798852884360f,    0.537824655594807f,
  -0.0767253125021830f,   0.444538451472890f,     0.623473048970629f,
  -0.500663871860675f,    -0.890399840538612f,    0.389528755348857f,
  -0.915832255765501f,    0.000652855725217894f,  -0.121310443088642f,
  0.206662014558968f,     -0.409513641801496f,    -0.0496262665388731f,
  -0.313314447256644f,    -0.994839397423865f,    0.344513198428247f,
  0.250828855150578f,     0.845438302422055f,     -0.728803841305459f,
  0.249670562418639f,     0.543601559270672f,     0.0138774767713057f,
  -0.0667600054234216f,   -0.803421294778238f,    -0.222729734665659f,
  0.461896933387103f,     -0.378537171475208f,    -0.464200027877777f,
  -0.363170335357481f,    0.616070694104851f,     -0.316407896795124f,
  0.131719997218670f,     0.0622146037260092f,    -0.881713850066484f,
  0.400811652868418f,     0.163777537634682f,     -0.528768052383715f,
  0.553072310703894f,     0.931393033749660f,     0.410062835546529f,
  -0.190904471223264f,    0.0533617852685424f,    -0.911780226731855f,
  0.823696403963215f,     0.756735978125573f,     -0.849701310148249f,
  0.106070214350541f,     0.747890454578944f,     -0.559823302095172f,
  0.976181619002882f,     0.506524051225122f,     -0.0735228576098872f,
  0.635610640336510f,     0.607728217052133f,     -0.383443012662118f,
  -0.640835123345673f,    0.0897243696426577f,    0.722421963278953f,
  -0.368833835044170f,    0.684790387373836f,     -0.0336846755494535f,
  0.199819176553169f,     0.351822803019512f,     -0.433387005248570f,
  0.709401898386598f,     -0.0149217994364210f,   -0.549115733466769f,
  -0.774049259429836f,    0.440376751789406f,     0.740171176715015f,
  -0.322301969056869f,    -0.148261856544327f,    0.724527166150266f,
  -0.744178178219827f,    -0.743031462890542f,    -0.00997727490160383f,
  0.550074849063942f,     0.147825200269716f,     0.777182602759074f,
  -0.625412073440604f,    -0.0614214671235789f,   -0.400121310797195f,
  0.864511820640236f,     0.327656445569618f,     0.765838911283705f,
  -0.906185069285438f,    0.543656228031101f,     -0.527337383463707f,
  0.544932532036177f,     0.453966596910417f,     -0.422906847383216f,
  0.803455668330395f,     0.496651297123425f,     -0.254890927444284f,
  -0.940902660088963f,    -0.0691448074129200f,   0.0165534278793877f,
  0.510199004798987f,     -0.0286331020627788f,   -0.141471298460923f,
  0.872000980716430f,     -0.752995088893842f,    0.167696515625982f,
  -0.181673581299286f,    0.496236252387172f,     0.854022562040503f,
  0.388320660177419f,     0.499320363074588f,     0.173522726183149f,
  0.0334192536945390f,    0.631347719906229f,     -0.832803059709609f,
  -0.523826088751894f,    0.322557683663180f,     0.0263621365506006f,
  0.948982322858062f,     -0.253991680115490f,    -0.165970359640120f,
  0.331700483099733f,     0.808731855823033f,     0.159862831431822f,
  -0.438178259673022f,    -0.943749594272300f,    -0.967819867274861f,
  0.263403865531262f,     0.710981741513574f,     -0.274597382335371f,
  0.929606564147885f,     0.125943272920181f,     0.691306164809532f,
  -0.607946869004681f,    0.284352421048012f,     -0.421663515398071f,
  -0.409479725854699f,    -0.152265311389352f,    0.630868673855242f,
  0.123144840061153f,     -0.645105689918733f,    0.360153393247973f,
  0.683885744053582f,     0.752598814717991f,     -0.581494857182821f,
  -0.469116962448560f,    -0.0691726199196117f,   0.174679188611332f,
  0.351269328558955f,     0.394815335607621f,     0.710281940645013f,
  -0.618593505217632f,    -0.721546422551907f,    -0.974088703589852f,
  0.939556772536401f,     0.599407011070674f,     -0.342213391542906f,
  -0.387135346574836f,    -0.572027944718123f,    -0.622717582512866f,
  -0.676949872287677f,    0.993953153886700f,     -0.784539234625462f,
  0.788778188174951f,     -0.0652679971583152f,   -0.988740647590182f,
  0.748989697777310f,     0.412949190397683f,     0.206661198525718f,
  0.573116044772809f,     0.938498079842984f,     0.743167714677278f,
  0.755679122637903f,     -0.295095987460132f,    0.217166189740252f,
  0.230160404687938f,     -0.504654557405015f,    0.472402206737240f,
  -0.867751757044285f,    0.869050101160567f,     -0.905285205825199f,
  -0.0698843699947245f,   0.762379282963140f,     0.634191197174691f,
  -0.498487028811837f,    -0.284257632541078f,    0.224245853978976f,
  0.412950901773606f,     -0.831984679101472f,    -0.375663639002356f,
  0.153699995838016f,     -0.953997055484851f,    -0.545360745186449f,
  0.637687001020610f,     0.465459355638311f,     0.0769011654935299f,
  0.267123343048604f,     0.545842501706277f,     0.778890986545214f,
  -0.363432183057524f,    0.479786652022207,      -0.600912698239979f,
  -0.738845504293020f,    -0.775987143750184f,    -0.705559714187038f,
  -0.310523750352236f,    -0.576081829930414f,    -0.0341897834633795f,
  -0.388414434291246f,    -0.790681299048144f,    -0.169440674711419f,
  0.219815472280053f,     -0.323451599202462f,    0.835623141427806f,
  -0.932446301638351f,    -0.831480966559550f,    -0.185050128422203f,
  0.946045240208487f,     0.864740749402213f,     0.916918979039328f,
  -0.204049261822351f,    -0.807183358636872f,    -0.484543897885746f,
  0.974235382435000f,     -0.208019257024664f,    0.647411336652954f,
  0.0961385231960816f,    -0.800258527388060f,    0.352982142334643f,
  0.917274278881503f,     -0.733934252997685f,    -0.229420044045673f,
  -0.358499183112933f,    0.469156578609832f,     -0.859359096702447f,
  -0.937762141277625f,    0.389776419837803f,     0.458425599271073f,
  0.542973137971009f,     0.675023236195573f,     0.944029213696263f,
  -0.774027667733194f,    0.262984845114612f,     0.842689106929982f,
  0.349251854560315f,     0.815938991679117f,     -0.226283690374971f,
  0.144356327986477f,     -0.610588223452142f,    0.539695204296007f,
  0.655759463021729f,     -0.725805170479948f,    -0.194977831685847f,
  -0.306105075607822f,    0.725461617920836f,     0.678283785172857f,
  0.250577882812283f,     -0.571672652704059f,    0.112132856850530f,
  -0.236412229648694f,    0.768173015701816f,     -0.799251028098975f,
  0.100723381526471f,     0.113856811781171f,     -0.0281630563735495f,
  -0.0727902548617043f,   -0.515248547261805f,    0.795765010992038f,
  0.505540143557856f,     -0.496124371632015f,    -0.363010091302494f,
  -0.302067159683438f,    0.941309812688142f,     0.0564765277142674f,
  0.733027295879568f,     0.582734217224559f,     -0.159007222603058f,
  0.827637470837748f,     -0.163060519537145f,    0.352357500273427f,
  0.920405360379926f,     -0.280691553157313f,    -0.401974149240862f,
  -0.131353114797667f,    0.0719728276882135f,    0.795795661384902f,
  -0.348203323368113f,    0.946184663961743f,     -0.188400643814906f,
  0.979319203447783f,     -0.132195434304746f,    0.585832597473452f,
  -0.894730397941282f,    -0.998045985412111f,    -0.717844040997160f,
  -0.706372640246558f,    0.237517748136224f,     0.767232946579208f,
  -0.246080656591091f,    -0.767887803661775f,    0.139501344992184f,
  -0.545658806327887f,    0.480755550666584f,     -0.355750609145607f,
  -0.493518864013929f,    0.832011102158605f,     0.122542855024589f,
  0.179356501845966f,     0.630805165349165f,     -0.888557403477561f,
  0.861375782261841f,     0.963467658712489f,     -0.00498707715217361f,
  0.341894517453263f,     0.654808049991043f,     -0.826909952854692f,
  0.101446328788119f,     0.401514152845232f,     -0.830556985096328f,
  0.832187560444347f,     -0.657254039822149f,    0.0304197382717133f,
  -0.718462386339415f,    -0.592343549551534f,    -0.356333235896531f,
  0.674135547073730f,     0.606490641440102f,     -0.707328770155748f,
  0.0251846271186025f,    0.763024927861424f,     -0.258224600040528f,
  0.456384203436896f,     0.626482995304888f,     0.162353458245830f,
  0.964280614412026f,     0.869262296229816f,     -0.0659501568862260f,
  -0.712869755397848f,    -0.946968242335746f,    -0.852822740386429f,
  0.791522782900379f,     0.824530390150335f,     -0.369383609091590f,
  0.118366422602132f,     -0.713278848975255f,    0.549165545117801f,
  -0.00201102645336770f,  0.748955154405439f,     -0.173689412898754f,
  0.175162399203493f,     0.0819730422177463f,    -0.804833155982895f,
  0.972966530563786f,     -0.0614871820303859f,   -0.293463394754661f,
  0.885919261783643f,     0.498531250561504f,     -0.808874001349436f,
  0.364344357769432f,     -0.945616638616975f,    -0.285864129675031f,
  -0.0438177789332626f,   0.303981486324719f,     0.362653007142366f,
  -0.543157427730716f,    0.174551703296805f,     0.140105048664068f,
  -0.704163993684247f,    -0.647461975308389f,    0.831243960763754f,
  -0.364954329841192f,    -0.730289885595360f,    0.0119708019435723f,
  0.796338505809816f,     -0.227851954967331f,    -0.927330125804492f,
  0.0602265250934577f,    -0.485204061877453f,    0.198319346525046f,
  -0.529723177394882f,    -0.321493822700232f,    -0.839566193416413f,
  -0.187812484529161f,    -0.396142329367383f,    0.367600156667632f,
  -0.922657847865138f,    0.893508892950972f,     -0.504434314314017f,
  0.663184814192863f,     0.887813887366393f,     0.267103483259066f,
  0.984313142773772f,     -0.667515321448428f,    0.0718416862496054f,
  -0.733363156570869f,    0.00186343206374962f,   -0.316531364321301f,
  -0.467549697367438f,    0.569865535259013f,     -0.556502178434536f,
  -0.650896672234238f,    0.564462797319346f,     0.585276582729153f,
  -0.433005641153548f,    0.847012427243871f,     -0.462088105064984f,
  -0.379468633087939f,    -0.0104892833799723f,   0.654191676584918f,
  -0.893278846859767f,    -0.689350274835588f,    -0.333220721049179f,
  -0.0461703436190983f,   -0.463411501818667f,    -0.995085073808794f,
  0.526075522777196f,     -0.0686703698159610f,   -0.855908120278260f,
  -0.239774384006192f,    -0.524142243888286f,    0.119526621106050f,
  -0.838266471869898f,    -0.459366707886497f,    -0.974921205300089f,
  -0.680517660007036f,    0.507695286553230f,     0.0920009889477380f,
  -0.674459855090400f,    0.554585280302756f,     0.357871391273056f,
  0.453052004120624f,     -0.991707675828263f,    0.144725488641274f,
  0.0886535789688503f,    0.708257184179799f,     0.579351194763774f,
  0.902098539548710f,     0.0104715251706708f,    0.112677648152527f,
  0.0513772996762050f,    -0.647561525299580f,    0.321958856072156f,
  -0.433510239079594f,    -0.481493822802105f,    0.651663699618654f,
  0.922649363108760f,     -0.751799312011289f,    -0.0336105332513619f,
  0.236872038257485f,     -0.0434863841224971f,   0.150810692021768f,
  -0.217629544451037f,    0.345890414626050f,     -0.471941673338326f,
  0.675001035054686f,     -0.986585320322202f,    -0.784679789758475f,
  0.270727429189404f,     0.595792127677512f,     -0.485969146811564f,
  0.222507692419212f,     -0.850070310429306f,    -0.575184466843042f,
  -0.220860571657717f,    -0.749449040845746f,    0.743039624335149f,
  0.463892797640518f,     0.224829531690830f,     0.935410439714992f,
  0.00609595972560872f,   0.830877831388658f,     0.0270299847557276f,
  -0.648763861115704f,    0.471982277585509f,     -0.145722971031426f,
  0.650947186397952f,     -0.266164907037466f,    -0.962378355156458f,
  0.354855353373398f,     -0.184127215272909f,    -0.825621979621661f,
  0.595495186093792f,     0.448679578752395f,     -0.839671989567806f,
  0.302158874138200f,     -0.735484620769119f,    -0.891040803749876f,
  0.880298595525880f,     -0.281199581528421f,    0.0195033020490396f,
  -0.511515485794419f,    0.447303195702203f,     0.375317547074287f,
  0.964442757731427f,     0.167643569291013f,     0.0118587246816413f,
  0.958187068873858f,     0.315395458761821f,     0.188852872643367f,
  0.417450657662866f,     -0.540566147670448f,    -0.422709015019828f,
  0.101425586029329f,     -0.235465301656357f,    -0.806044548641562f,
  -0.617153815671298f,    0.350658348898447f,     -0.738540593521098f,
  0.291893065415692f,     0.335435501842245f,     0.832048727909480f,
  -0.609539777284250f,    -0.436992256701542f,    -0.685315947977391f,
  -0.502107715051164f,    -0.893460699283628f,    -0.262263680492396f,
  0.454417031133778f,     0.223227655510993f,     0.605288383003966f,
  -0.698800586984034f,    0.864843125666124f,     0.363752223710394f,
  -0.354571459375900f,    -0.575008718239530f,    0.423061550052490f,
  -0.272459660313524f,    -0.116932919064239f,    0.547073367599225f,
  -0.890822451422250f,    -0.884262586749836f,    -0.889803003239001f,
  0.217660629852574f,     0.154863581361214f,     -0.333284425759330f,
  -0.826087281020982f,    -0.958198419703014f,    0.850114828540176f,
  -0.391190814837661f,    0.956578087128909f,     0.0541599967910713f,
  0.0988550815990206f,    0.851903747125444f,     0.361959550717838f,
  -0.901818125706440f,    -0.0561477675277424f,   0.522090821863134f,
  0.263383912024089f,     -0.161061362097086f,    -0.983707460720128f,
  -0.333128836619106f,    -0.546535222349413f,    0.627261888412583f,
  0.408731616102241f,     0.754700916401496f,     0.869772826180715f,
  0.362242883540519f,     0.853587698951791f,     -0.698910717068557f,
  -0.671945256263701f,    0.802655941071284f,     0.338701009518668f,
  -0.0297818698247327f,   -0.881311052338108f,    -0.296717226328950f,
  -0.965699941652671f,    -0.737164428831818f,    0.00804554422537485f,
  0.989716933531351f,     -0.832438692682457f,    0.454553001515962f,
  -0.933801685729775f,    -0.644562445615081f,    0.104508389084640f,
  -0.535426180524709f,    -0.937041822784313f,    0.599911275476691f,
  -0.789109397888652f,    0.821293320968620f,     0.818032308067912f,
  -0.838306491947354f,    -0.172883985566904f,    -0.185775969502745f,
  -0.672256019841514f,    -0.412525056012874f,    0.142272136963196f,
  0.792136721788200f,     -0.726314486042219f,    -0.445981475954073f,
  -0.857821372905156f,    -0.783006950965519f,    0.438776336055643f,
  0.400193156140386f,     0.177525578340235f,     -0.435380642286229f,
  0.547815754835977f,     0.0496394855194708f,    -0.442174406426496f,
  -0.0856142956982360f,   -0.0247840885457120f,   -0.779016166389253f,
  -0.511802368745331f,    0.319887353303028f,     0.721806644023428f,
  0.770423389111803f,     0.809969588377187f,     -0.196191981856391f,
  -0.105718971622809f,    -0.301674515042257f,    0.613622254387482f,
  -0.969517273103490f,    0.0144576310630131f,    -0.668829420461301f,
  0.750377960820232f,     0.696858494013122f,     -0.563485511352760f,
  0.726226115587466f,     -0.227540741583116f,    0.665488592033944f,
  -0.124611809537824f,    0.489550286613580f,     -0.579185308695604f,
  0.628687311174276f,     -0.295770837727116f,    0.240358361854250f,
  -0.155642183802961f,    -0.885945841456110f,    0.388592282428421f,
  -0.663862196774143f,    0.363779469451472f,     -0.371285870971327f,
  0.563159689631810f,     0.102725415308920f,     -0.320909176496511f,
  0.334328794247963f,     -0.401664407219370f,    0.726728495517480f,
  -0.192310060924823f,    -0.107973316004269f,    0.898177814643418f,
  0.456682306673978f,     0.890742303266606f,     -0.742770990765425f,
  0.0337493848747046f,    0.786190819119190f,     0.911503487800545f,
  0.288384155888888f,     -0.249479393879906f,    -0.431949793185094f,
  -0.0847659302921913f,   -0.475416985100444f,    -0.362720571751962f,
  0.676910741300893f,     0.00488530543559529f,   -0.227678010632002f,
  -0.0632947771540859f,   -0.990261099329279f,    -0.708485805011827f,
  -0.304846597458441f,    -0.480289782580152f,    -0.593254971635338f,
  -0.656335976085053f,    0.584373334310954f,     -0.493268395245234f,
  -0.00212668034894836f,  -0.480221591678953f,    0.622365041709782f,
  -0.258845071515928f,    0.943825418665593f,     -0.716642329101759f,
  -0.765317239111819f,    0.324487844009035f,     0.108158868464706f,
  -0.790583201992229f,    -0.649218622127061f,    0.751409704126257f,
  0.301455204388007f,     0.620482350165047f,     0.411016780608874f,
  -0.878843779367281f,    -0.779673415191805f,    0.616508572699874f,
  0.0750844738292273f,    0.341011338533919f,     -0.553376665552953f,
  0.277561087965059f,     0.527499935800293f,     -0.489644680144407f,
  0.514353996113782f,     0.229842524701725f,     0.139172928186734f,
  0.793753206591897f,     0.835555341130211f,     0.794120687009671f,
  -0.0994745468343306f,   0.109098970584400f,     0.383123470993648f,
  0.272549010931094f,     0.683070582699418f,     0.522823199313615f,
  0.235903759158310,      -0.269490013000195f,    -0.103775744391749f,
  -0.994083979953753f,    0.754983594207459f,     0.806308398378106f,
  -0.997543362839150f,    -0.00396367603607373f,  -0.873768378178592f,
  -0.755907732827809f,    0.703713206520365f,     -0.0716773056166142f,
  0.0792968663717508f,    -0.113760825029016f,    0.828188140127672f,
  -0.103062543982628f,    0.0455017026983378f,    0.330658414568756f,
  -0.615810862221588f,    0.827890015477212f,     -0.507551960954374f,
  -0.371044788092612f,    0.723489294741891f,     0.169072478802524f,
  0.885612989356318f,     -0.496475905980558f,    0.114400438991609f,
  0.427961880327008f,     -0.0456714004002505f,   0.0246660859589438f,
  0.175616122301987f,     -0.349777838484285f,    -0.939474935533562f,
  -0.215061649130134f,    0.907049169335834f,     -0.0553600192559760f,
  -0.982464152311714f,    0.405919915647442f,     0.755952405091542f,
  -0.695422520039876f,    0.373280568864688f,     0.483909023765611f,
  0.784896384994620f,     0.978722132488262f,     -0.113866140463085f,
  -0.630016943176703f,    0.512742627309861f,     -0.829104067044703f,
  -0.240982431155520f,    0.0107361024967163f,    -0.438682584788413f,
  0.935730031472303f,     -0.953447901200043f,    -0.984218956474073f,
  -0.745077052885218f,    -0.466232938128846f,    0.0326267564209573f,
  0.303877586274065f,     -0.199843777507458f,    0.674317529952029f,
  0.448678903834397f,     -0.681863209154081f,    0.273397524216090f,
  0.193101955704959f,     -0.342858479278718f,    -0.485179713360910f,
  -0.586067050491890f,    0.393099777352274f,     -0.982324485510343f,
  -0.852553426343700f,    0.773613825101220f,     -0.590256032959421f,
  0.837952413540589f,     -0.643137731235821f,    -0.311955662956384f,
  -0.888588599835619f,    0.304629859477166f,     -0.810098957400030f,
  -0.534291626181040f,    0.878601703692302f,     0.362706441157764f,
  -0.254447668911795f,    0.604309282304246f,     -0.977266419340276f,
  0.250927873824064f,     0.549600558999971f,     -0.796155833245480f,
  0.226373301058549f,     0.0137578302483823f,    0.819708534464965f,
  0.185662636424304f,     -0.450456459548662f,    0.0953849597308440f,
  0.736872088617975f,     -0.582024306116842f,    -0.0522261513001507f,
  0.394348349710790f,     -0.461023913227183f,    0.139996201153565f,
  -0.790168851966909f,    0.692544084408690f,     -0.580603732841955f,
  -0.584540773580447f,    -0.967062276813525f,    -0.00886260208554912f,
  -0.0520831218167985f,   -0.999614949922684f,    -0.965820736077636f,
  0.366390034326646f,     0.0323069925013668f,    0.164651515113853f,
  0.300260003499445f,     -0.340634856317630f,    -0.238157231550037f,
  -0.291645957143165f,    -0.773881882387456f,    -0.144494053860223f,
  0.660329619628580f,     -0.626727996257997f,    -0.994965090982706f,
  0.161018019917379f,     -0.327211572176153f,    0.0410991278573425f,
  0.0123663905917732f,    0.747176159655312f,     -0.485981637435718f,
  0.00667961234248971f,   0.631625759154389f,     -0.831294487064668f,
  0.449606477050286f,     0.768845094514142f,     0.928354534843426f,
  0.812647997969340f,     0.353418126917875f,     -0.872184763557736f,
  -0.579130598386915f,    -0.912928075675835f,    -0.779484407508668f,
  0.534916834944041f,     0.326353225230543f,     0.395431557674662f,
  -0.842103899863317f,    0.196590107332985f,     -0.261317824893025f,
  0.750190543523333f,     -0.103409967857074f,    -0.201452426430379f,
  -0.213633615009587f,    0.578822104214576f,     -0.130809161238349f,
  -0.774608769872343f,    -0.0222201705228122f,   0.126990738965544f,
  0.785780586747108f,     0.0379484317527632f,    0.837140835706189f,
  -0.191007948387153f,    0.106781679021568f,     0.990298140861558f,
  0.618337701073777f,     0.460255491901774f,     0.716379796730692f,
  -0.159421014009881f,    -0.560212468621569f,    -0.147263014783522f,
  -0.962301694075771f,    -0.327702010262213f,    -0.773959532468388f,
  0.351239668535113f,     -0.682281479449518f,    0.342188824054257f,
  -0.743039216419066f,    0.700710268270439f,     0.919651386092770f,
  0.626343233048871f,     -0.157189576636596f,    0.781882574006976f,
  0.349953565654219f,     0.361235312853466f,     0.313242228422046f,
  0.582185182102266f,     0.554504358491139f,     0.711217954194576f,
  0.332473377627418f,     0.165078226255772f,     -0.228349029389292f,
  0.899730713958153f,     0.653894503448836f,     -0.0452904440925501f,
  0.0328806142413372f,    0.793701315832839f,     -0.703826261467540f,
  -0.901648894320192f,    -0.195631966969018f,    -0.0470590812056508f,
  0.487185699934959f,     0.175961644103331f,     0.818028721319245f,
  -0.224389104974946f,    0.901974203693823f,     -0.153212477843726f,
  -0.472747796173897f,    -0.587471692952684f,    0.452340198339707f,
  0.996443894349412f,     -0.849126217374502f,    -0.403800337277983f,
  0.923427876645159f,     -0.0516037992113898f,   -0.380335341989182f,
  -0.299914673109747f,    0.764492139190834f,     0.773463290027243f,
  0.0175454601261817f,    -0.400742340353541f,    0.912354892189422f,
  0.999766609328281f,     -0.521321752061712f,    -0.365769506846305f,
  0.477612405338644f,     -0.0522578739905535f,   -0.479259238587280f,
  0.645161410912429f,     -0.702546085166056f,    0.359736398041538f,
  0.638130894056863f,     0.115633419893101f,     -0.674410360620500f,
  -0.150824943737990f,    -0.824854463897591f,    -0.504410162129685f,
  0.560317574021813f,     -0.159611666752889f,    0.997647540626334f,
  0.702777895178414f,     -0.946494281691535f,    -0.0109619562916898f,
  -0.383756482005404f,    0.872670066971334f,     -0.453527506439184f,
  -0.635719199113957f,    0.932852122005178f,     -0.800755479140234f,
  -0.225213334363716f,    0.251163542389519f,     -0.598147625383133f,
  -0.155241293946661f,    0.967736510890644f,     -0.0157250628103103f,
  0.250570924071858f,     0.209749651169078f,     -0.381016062687537f,
  -0.679300447230592f,    0.160197663113971f,     -0.749803147200800f,
  0.596917045783617f,     -0.0878737681749431f,   0.642402180339789f,
  0.261614973684270f,     -0.111833224093973f,    0.300170844971678f,
  0.317966800167647f,     0.0585375534708252f,    -0.842709435910728f,
  0.760207701069839f,     -0.979366191145221f,    0.940703569377911f,
  0.866488078693979f,     0.553497107695259f,     0.127260247084497f,
  0.530106152060111f,     0.725171359852920f,     0.356742729430045f,
  -0.209841680046178f,    -0.164239817187855f,    -0.888858150931758f,
  0.0367561852378047f,    0.803496113779956f,     -0.594927045375575f,
  -0.00347281985657166f,  0.114118941713783f,     -0.427864462568672f,
  0.719021423892768f,     0.335845790828654f,     0.0207216235296064f,
  -0.523146933862102f,    -0.145001077781793f,    0.490566784879983f,
  0.461904660734682f,     -0.897010089735077f,    -0.895737903861849f,
  0.343397505472310f,     -0.684377591381862f,    -0.0154016881290400f,
  -0.462987614871549f,    0.884045010701589f,     0.192617174725234f,
  0.226497290324550f,     -0.788151335932529f,    -0.190538526746651f,
  -0.556614046330326f,    -0.139480186854974f,    0.196785300148418f,
  0.978844132512627f,     -0.290726060479808f,    -0.591813978495167f,
  -0.0769033757443105f,   -0.467044929381376f,    0.171585053083057f,
  0.408215527269010f,     -0.818706013465989f,    -0.328144984930982f,
  0.790275356337217f,     -0.977491163139178f,    -0.979679268318504f,
  -0.524875121608236f,    -0.263859024168277f,    0.0180787743488171f,
  -0.984390626106750f,    0.952274619010224f,     -0.851400664579601f,
  0.692959439369046f,     -0.150312001943653f,    0.712066554169562f,
  -0.492336226254660f,    -0.453559897031351f,    -0.159679763180474f,
  0.745834647687870f,     -0.725963425297178f,    -0.720341794596050f,
  0.370674334928492f,     -0.845974926208293f,    -0.00448769398027360f,
  -0.595973105115042f,    0.967372249596385f,     0.512949503724102f,
  0.889619262804735f,     0.990718232652913f,     -0.662246751886904f,
  0.333846293708563f,     -0.423114421367372f,    0.549637439543149f,
  -0.987876053136374f,    -0.782714958794276f,    0.294868983681807f,
  0.931284560597614f,     0.445522387300861f,     -0.388400162488578f,
  -0.182673246109423f,    -0.773488958971573f,    0.438788569593725f,
  0.578106509978236f,     -0.373449127435319f,    -0.301996528814967f,
  -0.227124771031239f,    0.700176189695036f,     -0.910948938567526f,
  0.733412403327578f,     0.486154072292544f,     -0.974058632864456f,
  0.216693355653246f,     0.147564301397678f,     -0.715192277853558f,
  -0.366996833259925f,    0.568909126406069f,     -0.0810069456450131f,
  -0.371253841044151f,    0.254736918036059f,     -0.868966383080701f,
  0.190312518076662f,     0.457253801337437f,     0.941043431633233f,
  -0.297470749600241f,    0.244270515950156f,     -0.240122562119888f,
  -0.766384662307300f,    0.765045432900429f,     -0.608250739173787f,
  -0.733052557932594f,    -0.268433000443065f,    0.733598123424154f,
  -0.0550005774741753f,   0.273893221740822f,     -0.659641650983149f,
  0.967032725204337f,     0.390126626361090f,     0.518740746381756f,
  -0.859387560527806f,    0.554117289841284f,     0.648904904654236f,
  -0.755880975555381f,    0.834231592524942f,     -0.137512395743275f,
  0.0477027353535724f,    -0.880364563062979f,    0.458763614093086f,
  0.650036413308116f,     0.496385905878033f,     -0.418537115548864f,
  -0.565561960782851f,    -0.227941684691245f,    -0.165031891659812f,
  0.204464989908300f,     -0.688093624763916f,    -0.678874848552394f,
  0.813764873880514f,     -0.561723359541237f,    -0.575805702297063f,
  -0.288097000970518f,    0.950119107184838f,     0.709879842972902f,
  0.730067219897393f,     0.710813066057284f,     -0.192333836978130f,
  -0.190446300563246f,    0.872679304648751f,     0.134143163657763f,
  -0.979443835407234f,    -0.103872104041761f,    -0.0568328979324004f,
  -0.863020133862081f,    -0.0257801722427251f,   -0.577962771617033f,
  -0.0500056799801032f,   0.191817418291914f,     -0.799853775410853f,
  -0.110019424741421f,    0.840753817223395f,     0.355588322976119f,
  0.274501278628024f,     0.757538306972136f,     0.771547320156202f,
  0.0394143752709530f,    0.120744072764658f,     0.324337882930581f,
  -0.380086709776951f,    -0.772025774284869f,    0.473986846199588f,
  0.703247561676381f,     0.734667480205300f,     -0.594290184210087f,
  0.760158653782445f,     0.624553744314883f,     -0.941053266957965f,
  -0.165913770936962f,    -0.0497972870738055f,   -0.0435608680908517f,
  -0.663165366083943f,    -0.570972482385751f,    0.427845034528880f,
  0.0897903148165149f,    -0.481825010950428f,    -0.0901127105939594f,
  0.887770435656611f,     0.770985476674623f,     0.00966158758316293f,
  -0.331059327378268f,    -0.286033645163736f,    -0.0698945910210471f,
  0.834392309773299f,     0.875537383319608f,     -0.657919190548359f,
  0.583890957562885f,     -0.418481077359384f,    -0.282242397022386f,
  0.864577023994874f,     -0.898367126143440f,    0.815804441243808f,
  0.616061408588373f,     0.132365642864798f,     -0.221099752471970f,
  -0.852722283680675f,    -0.269499596712950f,    0.360828136129415f,
  -0.120022743070141f,    -0.0354652134632905f,   -0.718389836602256f,
  0.973490047219112f,     -0.201775047168341f,    0.348769511760972f,
  -0.338750368577880f,    -0.269769414088757f,    0.498910931428472f,
  -0.787648791515347f,    0.508408064858444f,     -0.904215976374529f,
  -0.778575029821227f,    -0.662889546847757f,    -0.787503064261069f,
  -0.915166838630178f,    -0.415784802770356f,    0.731806835017609f,
  -0.903922155472207f,    0.0872811033112211f,    -0.452516774501827f,
  0.577942533813694f,     -0.200909337658770f,    0.866167939661793f,
  0.982552542055944f,     -0.332277333696961f,    0.201673960342839,
  0.881239812364993f,     -0.0293753746942893f,   0.0967170348490725f,
  -0.765573023404242f,    -0.179225339525953f,    -0.931530757069740f,
  -0.702596334762137f,    0.439106079307245f,     -0.469364154277323f,
  0.211063395888038f,     -0.245858633045693f,    0.936376071219385f,
  0.0334087380010875f,    0.0765265939183459f,    0.417091701258078f,
  0.962467059286170f,     -0.180698008768999f,    -0.129816441691123f,
  -0.833694435146788f,    -0.800582099046532f,    0.736376297618233f,
  0.0164704176688124f,    0.207462305741760f,     0.300555292898496f,
  0.777154212278295f,     -0.0804533056660695f,   -0.279940128908185f,
  0.203101811030871f,     0.447496959357837f,     0.508353359025257f,
  0.644333822521829f,     0.897259297488483f,     -0.675785952117501f,
  0.149337263319588f,     0.350953290584184f,     0.600296681944338f,
  -0.606098182955297f,    -0.418312129297725f,    0.792551232171214f,
  -0.944025948110651f,    -0.923106441737020f,    0.508989820072736f,
  0.101554011154237f,     -0.799369609980037f,    -0.229001813644938f,
  0.196367996268564f,     -0.634078446275840f,    0.267446716753553f,
  0.943765754688567f,     0.329924442019441f,     -0.898235312442524f,
  0.563592978494850f,     -0.976934293161001f,    -0.609744819901837f,
  0.498989633313589f,     -0.105680058480959f,    -0.400730747241191f,
  0.264919109340783f,     -0.313066735594123f,    -0.465399967597728f,
  -0.425123918113080f,    -0.609514085808810f,    0.916560800692384f,
  0.0173757138934230f,    0.147814399202503f,     0.594152503614559f,
  -0.145681097751433f,    -0.427232299718493f,    0.233460382614713f,
  0.337361272635241f,     0.376106438004541f,     0.900277274651600f,
  0.424547631957395f,     -0.710790444715071f,    0.0846761090154495f,
  -0.0122707338404220f,   0.119989812955904f,     -0.239774389963524f,
  -0.692300891031819f,    -0.735109129583214f,    0.802276300301071f,
  0.348982047806247f,     0.916302084278941f,     -0.0838164783829127f,
  -0.989134997097880f,    0.832909602224562f,     -0.701363449605445f,
  -0.150487976031971f,    -0.728594035984111f,    -0.144393031996783f,
  -0.458856761770637f,    0.733295441303064f,     -0.405608670768629f,
  0.522871610912813f,     0.468223399458939f,     -0.575139530810903f,
  -0.241684287862418f,    -0.499140599234906f,    -0.395586476697394f,
  0.692745485195348f,     -0.125142235859546f,    -0.342212246193052f,
  0.133841188490164f,     -0.539478395228865f,    -0.887973984329817f,
  -0.474033882236453f,    -0.837114132429830f,    0.773392302912611f,
  0.117697651876253f,     -0.461595011213406f,    -0.528669601602068f,
  -0.957799577987062f,    -0.468654423525192f,    -0.0602288998398475f,
  0.154553704272891f,     -0.422854231304259f,    -0.496136532114270f,
  -0.348154983723668f,    0.0576478341707483f,    0.542088962901856f,
  -0.0465812136931592f,   -0.280128217727361f,    -0.900695482510248f,
  0.525110685457899f,     -0.957266165874283f,    0.136490670826643f,
  -0.213221811269364f,    0.690040133288898f,     0.269408771473479f,
  -0.0488994830172422f,   -0.837526616586426f,    -0.289127052660601f,
  0.149325279006459f,     -0.694169700971401f,    -0.0230547571616897f,
  -0.368313297034846f,    0.344434270521740f,     0.859135365902404f,
  0.839336654691204f,     -0.511783987355355f,    -0.0349625753049687f,
  0.935857929664427f,     0.820032045433520f,     -0.0394079346656324f,
  -0.656352913407746f,    -0.874383371678169f,    -0.425836335156061f,
  0.208600154889275f,     -0.135596548598733f,    0.566430757256762f,
  0.820840891306264f,     0.735746624790780f,     -0.765482927015804f,
  -0.0195537720748045f,   0.606216172027628f,     0.436027839798869f,
  -0.609233580289002f,    -0.963547951222316f,    -0.575271468261977f,
  0.692873344771925f,     0.143031668657597f,     0.890157114774225f,
  0.762299295692265f,     0.653618249618643f,     -0.957258626549595f,
  0.521895225378123f,     -0.607922211531407f,    -0.956795748110572f,
  0.477633684273092f,     0.794301967670603f,     0.139753218894595f,
  0.371726372555490f,     -0.804791987531745f,    0.837080126047059f,
  -0.440992020192054f,    0.584986017638085f,     0.950442046050057f,
  0.613109120495913f,     0.633948971396891f,     -0.581246845000116f,
  0.730290176291093f,     0.599119212595240f,     0.120096101936515f,
  -0.144169383323758f,    0.930776406826440f,     -0.0209712926465206f,
  0.572995665695966f,     0.924623298379120f,     -0.751832867985678f,
  0.630196806059302f,     0.506634179395662f,     0.0388997263873157f,
  -0.311041196366299f,    -0.729049093325017f,    -0.918815504666740f,
  -0.103935429478766f,    -0.000623544124330300f, 0.102880227280474f,
  -0.563637096166535f,    -0.332148269587814f,    0.472114131256244f,
  0.295717126494164f,     0.246944592105312f,     -0.713191555660498f,
  0.160410320426559f,     0.110992936470077f,     0.213877527744528f,
  0.541660996543375f,     -0.872405734998843f,    0.388515073094269f,
  -0.840811647524440f,    -0.968008592072007f,    0.669947948420772f,
  -0.122943215855172f,    0.565929115911552f,     -0.695408966310186f,
  0.361296950635219f,     0.574282481983669f,     0.0877180513263536f,
  -0.694316083550519f,    0.327696487191071f,     0.289746985823208f,
  -0.241476327174879f,    0.605084742574250f,     0.0929272338220821f,
  -0.391399761658219f,    -0.612928183827531f,    0.0471987261466311f,
  0.157388702609590f,     0.575695018001234f,     0.450453491026024f,
  0.876623108212541f,     -0.456500163180038f,    0.436901006801809f,
  0.796734433864345f,     0.771008396172517f,     -0.784740610155705f,
  0.405172719255834f,     0.958393667559228f,     0.787380105147761f,
  -0.262826016234054f,    0.773327117333271f,     0.482142068916266f,
  -0.461607549022954f,    -0.153993688218026f,    -0.129280134980317f,
  0.901812622560630f,     -0.111520793491644f,    -0.0973214524989203f,
  -0.293695817178366f,    -0.190045093887485f,    -0.204792515844396f,
  0.501086384719391f,     0.755953359112033f,     -0.425886872154604f,
  -0.0883029298084141f,   0.763071371252921f,     -0.556289447935984f,
  0.577370462369201f,     0.0480599614417476f,    -0.794423686623353f,
  0.756645959967545f,     0.570538730848462f,     0.872575422156333f,
  -0.443572567528656f,    -0.0487937634747691f,   0.283986553648095f,
  -0.170910821134099f,    -0.329867000423004f,    -0.982322841409943f,
  0.555344201026651f,     -0.351964643393940f,    0.776172688776518f,
  -0.148102477734717f,    0.889532618676503f,     -0.310979434517253f,
  0.711839903052208f,     -0.646385596147085f,    0.145592596381502f,
  0.233949589173221f,     -0.825471565980294f,    -0.370248763132654f,
  -0.777194557275684f,    -0.224658064754195f,    0.263281286751478f,
  0.849661910468068f,     0.271261490445121f,     -0.915885420717958f,
  -0.947144520818678f,    0.227960459606299f,     0.784463828083640f,
  0.995882406349565f,     -0.987273766396493f,    0.0792453274840108f,
  -0.788403526960056f,    -0.619975942121645f,    0.801181796307713f,
  0.967884377026145f,     -0.781223064263388f,    -0.300716486479280f,
  0.994748932974184f,     -0.200152360574411f,    -0.101898131541608f,
  0.542914585925881f,     0.407729967031792f,     -0.105215843903154f,
  0.638066037611924f,     -0.563777780161298f,    0.134189395993685f,
  -0.503320561486155f,    -0.0379170306314711f,   0.723638115686875f,
  0.747948383928228f,     0.928239905995551f,     -0.736883772878758f,
  0.892242913709735f,     0.468998243295705f,     -0.224406388545097f,
  0.758754382878863f,     0.994739001052496f,     -0.749837906573089f,
  -0.938777322178786f,    -0.619168635741936f,    0.827875717654585f,
  0.294033159230782f,     -0.372766318349126f,    -0.292752124932124f,
  0.396629951868878f,     -0.986760927173237f,    -0.0841834195975009f,
  0.999760803826313f,     0.0142305924173638f,    -0.820393900206961f,
  0.409972278573230f,     0.227315924377402f,     -0.641500351639361f,
  -0.470788010535406f,    -0.486171076557593f,    -0.758140688442947f,
  -0.971539708794928f,    -0.949039718833189f,    -0.146844988902767f,
  -0.0183627478820223f,   0.402918762093981f,     0.0620266698060286f,
  -0.182527786403967f,    -0.374395326540229f,    0.566584207940253f,
  0.879546558847970f,     0.853360173786566f,     -0.515321950652696f,
  0.511692631053674f,     0.342152084355850f,     0.374686420595610f,
  -0.794372980760555f,    -0.648670375991101f,    0.761192158420166f,
  0.223791993225057f,     -0.342720055148453f,    0.965612513218950f,
  -0.796193009647904f,    0.215057114709867f,     -0.0459498239576994f,
  0.871047701509015f,     0.664672380241520f,     -0.546301701630944f,
  -0.939910946986200f,    -0.213195706858966f,    0.559543622118596f,
  -0.255844807516886f,    0.509576048776352f,     -0.699005089750431f,
  -0.520317652140772f,    -0.924306703712950f,    -0.923814193467638f,
  0.868401299001930f,     -0.571229497763863f,    0.984740691690212f,
  -0.911782692220985f,    -0.265295471266664f,    0.0479848731515942f,
  -0.195328058836883f,    0.758281465939343f,     -0.418177229854869f,
  -0.263323557662932f,    0.0230762644115943f,    0.382605016442608f,
  -0.576209059863238f,    -0.739785100410209f,    0.0956412509899256f,
  0.0369702493097637f,    0.0738922616872486f,    0.589371657036664f,
  0.548586250623500f,     0.996096574632666f,     -0.574178408335425f,
  -0.827059309028347f,    0.600283403682961f,     -0.0651062813338117f,
  0.985857002071398f,     0.982700721670305f,     0.777628710286989f,
  -0.139415722014730f,    0.951156387462424f,     0.806391217144736f,
  0.135433009714206f,     0.252388414319270f,     0.485541324740928f,
  0.270688932431637f,     0.892850103229909f,     0.440168171407923f,
  0.515384398158669f,     0.600884162546465f,     0.947986221531091f,
  0.339440884303404f,     0.403857490690436f,     -0.937015609644647f,
  0.729529495316627f,     -0.389601866986821f,    -0.420712615666380f,
  -0.763003723744745f,    -0.0619534667970105f,   0.486654476027536f,
  -0.943536854881494f,    0.471171699317719f,     0.996886209046820f,
  -0.945316270673373f,    0.230772742042993f,     -0.621222111022648f,
  0.838934157721328f,     0.124035987915113f,     0.737576768711407f,
  -0.217898078842006f,    0.0429859145211120f,    0.223685413947773f,
  0.820073956039170f,     -0.378381145423743f,    -0.335672684173821f,
  0.649791267584388f,     -0.457253860252872f,    -0.664776842833046f,
  0.150429615666837f,     0.974812973893170f,     0.00119972362050369f,
  0.140744912838368f,     -0.252632269055503f,    -0.124205752907507f,
  -0.383194456927254f,    -0.356455432479067f,    0.0694989880525767f,
  0.188230048541949f,     -0.854592697407303f,    -0.902559387772971f,
  0.454054169179423f,     0.534684767654295f,     0.806837289706952f,
  0.274203715752641f,     -0.765433763984323f,    0.459365005291520f,
  -0.896797218412250f,    0.382900474341852f,     0.169400421233177f,
  -0.184111368111075f,    0.0323514487432812f,    0.621015577938758f,
  0.139872518806323f,     0.480965263781330f,     0.0649386999855643f,
  0.815365754221614f,     0.761990264098834f,     -0.0927412249348933f,
  -0.580853742457387f,    0.211615321410605f,     0.165159968106305f,
  0.305515629345863f,     0.725748395743965f,     -0.667649812347274f,
  -0.621843189978885f,    -0.939317191264789f,    -0.197108718554958f,
  0.902152006895939f,     -0.889744652803018f,    0.667113256888905f,
  0.929471703711725f,     0.660025836042506f,     -0.0712223078913006f,
  0.416152292126436f,     -0.602223852277700f,    -0.828462878627106f,
  -0.956915163338265f,    0.298196541541469f,     -0.933863927954050f,
  -0.198745190221695f,    0.749101206471011f,     -0.922366396086261f,
  0.769953026855636f,     0.971459582749177f,     -0.226637139032289f,
  -0.593509265485619f,    -0.635649447577657,     -0.443127775644156f,
  0.350464269654307f,     0.379979516655134f,     0.896282784801247f,
  0.00871209446887344f,   0.401818712823609f,     0.815422566258939f,
  0.215868289995843f,     0.682217845700443f,     0.508819667108007f,
  -0.988484263336122f,    0.216656890144568f,     -0.185777888700071f,
  0.522106353117928f,     0.894211314619113f,     -0.779300881699217f,
  0.137801937535128f,     -0.818740955579722f,    0.637214461095055f,
  0.187867696634722f,     0.184985729971243f,     0.315323389557324f,
  -0.0312525033775366f,   0.498559120407008f,     0.855778208118391f,
  0.936851170962385f,     -0.0524308158818188f,   0.257087262622978f,
  0.816141818246927f,     -0.147192803443011f,    0.194545158383538f,
  -0.655428449892669f,    -0.650441844539509f,    0.536015423540886f,
  0.0250060573607953f,    -0.863380305825989f,    0.0605420782823460f,
  -0.963662464088496f,    0.689136717877590f,     -0.929664162821947f,
  -0.327349437742288f,    0.713122240487331f,     0.765587094162777f,
  -0.314350325341316f,    0.409992519686522f,     0.753377832105546f,
  -0.756848529995586f,    0.760787899507869f,     0.512213162407276f,
  -0.674820237484644f,    0.560776719592082f,     -0.874905891603855f,
  0.925202682923872f,     -0.907405002733482f,    -0.575836335118172f,
  -0.248173888600965f,    -0.187923239740639f,    0.230951002247789f,
  -0.540190666146588f,    0.390890663320481f,     -0.705511708249712f,
  0.0980457138183717f,    0.879979753648798f,     -0.378326046226794f,
  -0.645363625221967f,    0.883365508962968f,     0.728720763588748f,
  -0.191571576393619f,    -0.941989254130187f,    0.944312154950866f,
  -0.367184985473008f,    -0.974124559264444f,    -0.579946765132286f,
  0.509825236656578f,     0.952047194261820f,     -0.0955445631918663f,
  -0.00500764501201401f,  -0.00111382665477655f,  -0.0404281661495578f,
  -0.265706359102834f,    0.865881843285797f,     -0.947915521623861f,
  -0.820337973623839f,    0.0843747524022067f,    -0.948599514028391f,
  -0.464018526769358f,    0.600790429663803f,     -0.0779017384430381f,
  0.756949801920938f,     -0.955436496929340f,    -0.553346424499498f,
  -0.401256107066610f,    0.569624108543687f,     0.179455179577041f,
  -0.189386842296675f,    -0.467166492259358f,    0.367644583467601f,
  -0.722338735126514f,    0.863903729827081f,     0.0631027352569811f,
  -0.982269235503679f,    -0.837788470642698f,    0.421730643738386f,
  -0.671745211565315f,    0.858467932275763f,     -0.745298219348761f,
  -0.659594977600028f,    0.403238381269873f,     0.951987904652099f,
  0.228887404582426f,     -0.331665752024408f,    0.794789885033899f,
  0.669978127515269f,     0.977583870328654f,     -0.346398989178462f,
  0.692053246433782f,     -0.159407706019695f,    0.710808563527500f,
  -0.555701359319642f,    0.625665798239905f,     -0.711048329414687f,
  -0.672431532474912f,    -0.474897384314332f,    -0.196250611816064f,
  0.902140605659856f,     -0.459732035217428f,    0.651412290305649f,
  -0.686137550630920f,    -0.803228611526547f,    0.371120664039117f,
  0.289869860968561f,     -0.720979161638185f,    -0.0940498575417996f,
  0.185025844935128f,     0.401524077274769f,     0.811721346556136f,
  0.224659861626089f,     0.106438807548742f,     -0.117458956991326f,
  -0.407361487623449f,    0.683891165426988f,     -0.216582410631386f,
  0.710644530504861f,     0.867797453793643f,     0.626683550176870f,
  0.115061097783331f,     0.976742668387085f,     0.250700864990527f,
  0.272723539841862f,     0.159923684669346f,     0.167713264013185f,
  -0.445764377935606f,    -0.489538472614810f,    0.227880894824940f,
  0.670702116476237f,     0.610361511284318f,     0.503801949624464f,
  -0.687816091694902f,    -0.0413765153535617f,   0.155769004545734f,
  0.921910233366689f,     -0.467299226678025f,    -0.991984541712805f,
  -0.527009262324220f,    0.248157897392517f,     0.661145853979517f,
  -0.975947426744844f,    -0.242453990684693f,    -0.277956284573619f,
  0.162010437415540f,     0.889456199489152f,     -0.171259539670729f,
  -0.0636124576727060f,   0.311318764402696f,     -0.227771282875219f,
  -0.567702050585727f,    -0.132881625149059f,    0.870846950418812f,
  0.440078398779761f,     -0.0908818839265000f,   0.410077545060762f,
  0.917678125288724f,     0.975295290129489f,     0.736514272579886f,
  0.653896379317074f,     -0.166512942888681f,    -0.218665383726096f,
  -0.0688642360506688f,   -0.596589868100824f,    -0.180873413844075f,
  0.229002598511067f,     -0.647630976455599f,    0.722615884501717f,
  0.760194030884127f,     0.253262836539679f,     0.0734191803957118f,
  -0.941427952376035f,    0.224118866807764f,     0.634990976599086f,
  0.538622500570355f,     -0.591487367587299f,    0.829253069890529f,
  0.426659996899884f,     -0.562435396124737f,    0.924178169394878f,
  -0.693964899988321f,    -0.520472617448914f,    0.845157115508053f,
  0.162984246343684f,     -0.212032053476592f,    0.0482566706558292f,
  0.820584028875367f,     0.676120066619505f,     0.590174358812695f,
  -0.457289938956925f,    -0.351282540371674f,    0.322162683499620f,
  -0.683726196205246f,    -0.279636659553935f,    -0.186133028676429f,
  0.965481755833750f,     -0.0550172560314044f,   -0.437844829991532f,
  -0.448670532146325f,    -0.438916826946834f,    0.830205353164842f,
  -0.0125988502002286f,   0.733716462327519f,     0.870000673588185f,
  -0.189915082276716f,    -0.676269331249200f,    -0.336432931956768f,
  -0.288892891213265f,    -0.912569275291884f,    0.509853767908707f,
  -0.658452317958678f,    -0.562848133961047f,    -0.102082581799095f,
  0.904062026055565f,     0.473339990381854f,     0.210234896873676f,
  -0.0884007008398613f,   0.720872020499257f,     0.538315255331760f,
  -0.884485227439286f,    0.160844002639634f,     0.625863524205804f,
  -0.947487159926400f,    0.362643826956149f,     -0.189913270725334f,
  -0.110428721523612f,    -0.666510263156819f,    -0.214827103263521f,
  0.912669747474334f,     -0.973896103049543f,    0.665373714127588f,
  0.148135031012834f,     0.126524689644449f,     0.00283763548841764f,
  0.312700495893193f,     0.579520771033243f,     0.677583023476560f,
  -0.779567427807191f,    0.0694994546110597f,    -0.298762697062437f,
  0.655210050716681f,     0.435909078048151f,     0.322095567178671f,
  0.764827170021089f,     -0.713736794113842f,    0.992844460358584f,
  -0.735915506109616f,    0.280204875392391f,     0.584446532772711f,
  0.796955505835788f,     0.742508124239176f,     0.0785523490091065f,
  -0.562359397016753f,    0.874448473576734f,     -0.794251927759664f,
  -0.658767152705445f,    0.120015806343044f,     0.662372174700575f,
  -0.719334975225296f,    -0.663474261357014f,    -0.637663874969148f,
  0.706137632813821f,     0.734790814693796f,     -0.449118755654663f,
  -0.758125670003823f,    0.719059339327447f,     -0.228679956701166f,
  -0.0782671261690160f,   0.637830522744746f,     -0.178696376536345f,
  -0.848273935253246f,    0.840882430630200f,     0.977813953976437f,
  0.565474986185913f,     -0.807314895274907f,    -0.100534840844589f,
  -0.436186956483089f,    0.854663592026441f,     -0.547576146320248f,
  -0.621784076386717f,    0.688687549426321f,     -0.688962085987764f,
  -0.998914668418794f,    0.751493418398842f,     -0.203018738091861f,
  -0.881317097659280f,    -0.422480898609404f,    -0.321074554557095f,
  -0.759379357125740f,    -0.806503084491033f,    -0.496837315822352f,
  0.217087355208111f,     -0.776801484423500f,    -0.445747498145286f,
  0.710204776554782f,     0.274276964033182f,     0.650397224484409f,
  -0.709395921248168f,    0.862663541330686f,     -0.946166202558813f,
  0.826638502366159f,     -0.450587332736099f,    -0.808257632193740f,
  -0.414360554482101f,    -0.471118187583276f,    0.981592919290155f,
  0.192794908371370f,     -0.314979855997427f,    0.722518962804398f,
  -0.795914669179603f,    0.121447532644509f,     0.0446893237592363f,
  0.651720955387594f,     0.897128141094619f,     0.283834144643742f,
  0.369570391543943f,     -0.163784005163726f,    -0.799144231493300f,
  0.338136741961422f,     0.795991730702685f,     0.601735561139351f,
  -0.556654767533027f,    0.907044495725416f,     -0.374604065784494f,
  0.814308532452677f,     -0.254295412850351f,    0.443103437041340f,
  -0.0218296619602199f,   0.826728672505738f,     0.773205771668962f,
  0.171909022893217f,     0.497670481959597f,     0.954178712898056f,
  0.0840098577761919f,    -0.705861127301893f,    0.145663865959608f,
  -0.436204975766037f,    0.479359595998989f,     -0.719493824988072f,
  -0.523146212355768f,    -0.917822711649927f,    -0.610003715217602f,
  -0.192266667446473f,    -0.377507163265653f,    -0.250419291332051f,
  0.873627391381727f,     0.922899703740095f,     -0.902411671519496f,
  0.285830821349708f,     -0.577368595723736f,    -0.598296174995687f,
  -0.0152478418690674f,   0.503725955636280f,     0.946501779740920f,
  0.261108140547963f,     0.206258978593364f,     -0.887022338332430f,
  0.989187042741485f,     0.461764104690670f,     0.305280284664753f,
  0.243972878436235f,     -0.573704516784209f,    0.111805651228880f,
  -0.373590027525854f,    0.574564836347642f,     -0.712884790778729f,
  -0.0570130063179222f,   0.244209425500712f,     -0.717492787619277f,
  -0.476920207759357f,    -0.444169983027413f,    -0.254851417015366f,
  -0.505678630542571f,    -0.953549022234155f,    -0.0316841901798541f,
  0.198256779602804f,     0.151938229162240f,     -0.0259028503944394f,
  -0.799645893003010f,    -0.889308912372168f,    0.339221517072804f,
  0.904784479404768f,     -0.367330903112591f,    0.866281762131661f,
  0.112765232993802f,     -0.0852946527317187f,   -0.283538359072154f,
  -0.734951426632046f,    0.502970854898684f,     -0.541434927857400f,
  0.881496286285600f,     -0.227404039639917f,    -0.636983936776183f,
  -0.0799774217214970f,   -0.833780310813424f,    -0.222787370954425f,
  0.433143783060434f,     0.0953330524947187f,    0.965400264971588f,
  0.308927931247299f,     0.344316393259575f,     0.122880788538352f,
  -0.898509922382301f,    -0.187062523329053f,    0.705352247460646f,
  -0.817811000761718f,    0.303714513401701f,     0.714863075518907f,
  -0.00862372607283035f,  -0.842715848975590f,    0.816504077307885f,
  0.924594085591125f,     0.334618732730041f,     -0.212414743241377f,
  -0.758289625449925f,    0.586405661412351f,     0.909247363444287f,
  -0.800422609846793f,    0.397430897916299f,     -0.408827454151232f,
  -0.411913213123543f,    -0.602703152770135f,    -0.893591462026327f,
  0.417648762458765f,     -0.766362696266534f,    -0.166060103951854f,
  0.883234167729589f,     -0.0741908774062401f,   0.113912882075078f,
  -0.268248292164738f,    -0.825585719915457f,    0.885446166477969f,
  -0.996523379251940f,    -0.000841720632677401f, 0.940286529247477f,
  -0.528330498750176f,    0.0938880690147421f,    -0.966296878893937f,
  0.891956527154360f,     -0.483384605653306f,    0.257210342748458f,
  -0.820220338820906f,    0.363913841603935f,     0.0364865250689275f,
  0.0619156958713947f,    -0.645447937080250f,    0.548279343062761f,
  -0.289526240449473f,    -0.506780094171335f,    -0.901771170107367f,
  -0.437874075223813f,    0.748512212111141f,     -0.529884246718074f,
  0.924062132675193f,     -0.365432219122282f,    -0.263296006595835f,
  -0.927083881647913f,    -0.192737974697553f,    -0.450051159199964f,
  -0.543528645806642f,    0.834976909049276f,     -0.426975046433596f,
  -0.361056079272416f,    0.883880063360531f,     0.680380429911630f,
  -0.553642515320953f,    0.548847108935282f,     -0.357430246936948f,
  0.210445016993628f,     0.949511601115471f,     -0.611278947360487f,
  0.344744934459962f,     0.0684247970496175f,    -0.877154656281116f,
  -0.521992702610556,     -0.0303764312006813f,   -0.647220068176984f,
  0.693175336224119f,     -0.0955602614554496f,   -0.765579758912278f,
  -0.821318118938906f,    -0.220936603794347f,    0.159013709512021f,
  0.0222743973539492f,    0.569438412513281f,     0.896083437551563f,
  0.973699071868637f,     -0.403438951991928f,    -0.976931032127622f,
  -0.0720613180573018f,   0.0788813367661694f,    -0.430781354548607f,
  0.580378296309349f,     -0.175446689199481f,    -0.256743557012462f,
  -0.696667845393283f,    0.870473046831235f,     0.146660713923108f,
  0.277741407197705f,     0.502075064404417f,     0.396530064046844f,
  -0.000209092342246420f, -0.977003947244262f,    0.451457326960000f,
  0.420509664462095f,     -0.0826395067671402f,   0.461120688156973f,
  0.786867285802415f,     0.429254905841222f,     0.894426863739026f,
  -0.670297281923597f,    -0.833650409296060f,    -0.908588009702110f,
  0.516311115539149f,     0.975234001829324f,     -0.532953533466378f,
  0.775291582519158f,     -0.0136022600428900f,   0.654817093112596f,
  0.363512141498233f,     0.624779024037534f,     0.0237004661473674f,
  -0.172570506046968f,    0.401807838319043f,     0.997391258152958f,
  -0.553969395939123f,    -0.415425175833161f,    -0.758032843655304f,
  -0.482766088920005f,    0.637574309072414f,     -0.729000055114342f,
  0.699851428676091f,     -0.827508053421131f,    0.900655803848482f,
  -0.431149800814228f,    0.0369409101983413f,    -0.378608101457895f,
  0.237564147838841f,     0.533020461112441f,     -0.280269627508005f,
  -0.864065787343603f,    -0.0381481642453043f,   -0.566886547530062f,
  0.539727700361167f,     0.166859339425035f,     0.850080295054718f,
  0.384690146030125f,     -0.384995288415294f,    0.303656036600558f,
  -0.580297619022502f,    0.0649069482840878f,    -0.162922327392773f,
  -0.235019427063355f,    -0.265468718118809f,    -0.121827312187455f,
  0.0416628805824146f,    0.343481543012411f,     -0.251429566892972f,
  -0.868100204320718f,    -0.802636407512128f,    -0.549547579028752f,
  -0.570017983863503f,    -0.853634311513627f,    -0.564570567173235f,
  0.955944215494794f,     -0.0930750790375956f,   -0.160319122401953f,
  -0.640886790354213f,    0.798634607857513f,     0.503051518023559f,
  0.765247226736789f,     0.909476811674882f,     0.677590253114963f,
  -0.110641683440517f,    -0.336445241915220f,    -0.684064840782028f,
  0.962285048920031f,     0.883303701653897f,     0.981819291389659f,
  -0.597290928759656f,    0.215792997443025f,     -0.847656608719347f,
  0.679887992445640f,     0.299901700372808f,     -0.677306526467426f,
  -0.348340058872692f,    0.651490451411335f,     -0.133387041637395f,
  0.718311240322040f,     0.0869279817052975f,    0.155413706090559f,
  -0.869119988858735f,    -0.566773040844476f,    -0.0513826414151206f,
  -0.368087669232071f,    -0.978175512831125f,    -0.229213501073727f,
  0.344608572405871f,     -0.663307667219997f,    0.437238632879575f,
  0.00205230197288353f,   -0.0897076092856746f,   0.834529513214144f,
  0.131872357342232f,     0.113081940417244f,     -0.418620232731326f,
  -0.317993033651213f,    -0.740303025960662f,    0.423423655701288f,
  -0.300833032468860f,    -0.458960388256530f,    0.692670405117589f,
  -0.559944357561921f,    0.0168623577148430f,    0.568661331088367f,
  -0.385055363002398f,    -0.356055436463140f,    -0.794446573681063f,
  0.908870080953069f,     -0.295500656666577f,    0.800625150733729f,
  0.206307902542489f,     0.729591183391974f,     -0.0655746333947396f,
  -0.261707022686154f,    -0.802380330579914f,    0.0812359238243023f,
  -0.00528231140765212f,  -0.725740453383981f,    0.919076065030463f,
  -0.896497189839174f,    0.861919731820265f,     -0.804273875755869f,
  0.230339021648310f,     0.296779613186519f,     -0.349872572510143f,
  -0.270230381483447f,    0.0368924200249658f,    0.581340248642417f,
  0.943620537648739f,     0.715012058065301f,     0.528414993233909f,
  0.695917111744314f,     -0.634354198968852f,    -0.483786223099716f,
  0.565405035681248f,     -0.530076864213017f,    0.363019522302994f,
  -0.825556544716473f,    0.891096876998683f,     -0.990692760548295f,
  -0.450641405862313f,    -0.597008073985341f,    -0.464377765418678f,
  -0.942926913464693f,    -0.871399725569805f,    0.232335933943403f,
  0.858786794807406f,     -0.528589179815518f,    -0.324757177062634f,
  0.595880088750788f,     -0.976574570427974f,    -0.423824220654658f,
  -0.832990206908489f,    0.198704682807118f,     -0.168244652325292f,
  0.843066822744011f,     0.0912498543932607f,    0.485570815146582f,
  -0.104653316420662f,    -0.623461298489716f,    -0.807713596811018f,
  0.737782734425857f,     0.456364368166532f,     -0.430703367862900f,
  -0.188953991637209f,    -0.827984282695373f,    0.0246267653665548f,
  0.891225605267640f,     0.910600867999638f,     0.345236086687552f,
  -0.600682365520065f,    0.833182106437698f,     0.213749250288017f,
  -0.0866339102562885f,   -0.618385082289017f,    0.859527120927500f,
  0.749978780964161f,     -0.334770513867011f,    0.242140166670949f,
  -0.196268320459958f,    0.611789869603675f,     0.655057159657307f,
  -0.603759576722096f,    0.614654509385217f,     0.144145218488192f,
  0.959930150756613f,     0.485009777784726f,     -0.564230295010912f,
  -0.404716165405314f,    0.0442672151313601f,    0.929486639423805f,
  0.409386317338224f,     0.527053707674182f,     0.899087569745327f,
  -0.933259779365388f,    0.265159475034860f,     -0.858300862890810f,
  -0.870994388031662f,    0.354868177430506f,     0.00956840260511749f,
  0.429740959889133f,     0.649668163567379f,     -0.744532888765288f,
  -0.967499901569196f,    0.556703631745254f,     0.535130550118618f,
  -0.639502350153040f,    -0.604586469532735f,    0.0799683564329623f,
  -0.156074786599444f,    -0.348308700325411f,    0.217829052228100f,
  0.545642400171123f,     -0.303317700019152f,    -0.473220675222451f,
  -0.239688108834945f,    0.0998500862725149f,    -0.962734081833842f,
  0.870743993144299f,     0.464578557934316f,     0.184511089576136f,
  0.559729843314504f,     0.0702052363354577f,    0.632714874625648f,
  0.212930743289312f,     -0.454606863365109f,    -0.592679055778218f,
  0.287649993384466f,     -0.457293694071368f,    -0.423493046785686f,
  -0.0674763327876298f,   0.242131064298176f,     0.488581911885965f,
  -0.464567743213882f,    -0.387515661812354f,    -0.914585596974616f,
  -0.255803162310627f,    0.941267268311980f,     0.690278917089395f,
  0.302397314111962f,     -0.178461434689705f,    -0.949279941481428f,
  0.160440202901122f,     -0.970582196769486f,    -0.0119478205074164f,
  -0.206440255898676f,    0.221640403444713f,     -0.819801447827624f,
  0.263614394802488f,     0.616376195532700f,     -0.596859494305351f,
  -0.118659509995453f,    0.458168997595326f,     -0.0400474705134108f,
  0.934465050133603f,     -0.852936731989621f,    0.0191637795580570f,
  0.298534793677081f,     -0.857491630206749f,    -0.0141198383157879f,
  -0.365027350962024f,    0.450964838023674f,     0.351383095290905f,
  -0.387039947149600f,    -0.983994933095116f,    0.610531582220017f,
  -0.0446025524732094f,   0.216718014780746f,     -0.676819246943449f,
  0.0385619292249610f,    0.192482456707739f,     -0.288809653393521f,
  0.241774557042318f,     -0.444638770943313f,    0.535319194413803f,
  0.374773141606987f,     0.186364279454450f,     0.0701814972821988f,
  -0.452753172654203f,    -0.350918291268194f,    -0.332963791049667f,
  0.179301863965318f,     0.954101654404080f,     -0.687960044344130f,
  0.611454049205213f,     -0.696789567124132f,    -0.551566492897529f,
  0.656434797122885f,     -0.601779335396959f,    -0.265656331560395f,
  -0.528821434638507f,    0.153601151147409f,     0.514739334540489f,
  -0.0517769842323894f,   -0.659246830986894f,    -0.453055366696259f,
  -0.0515886000780059f,   0.958478845408115f,     0.0221452906045994f,
  -0.159960643390796f,    0.816263632871352f,     0.245244170325114f,
  -0.0919839688704780f,   0.947170598807362f,     0.846772793441790f,
  0.247105133025056f,     -0.801972939368103f,    -0.224977420586025f,
  0.130099925027197f,     0.497816036746753f,     0.308139730113712f,
  -0.0536876417759813f,   -0.492022090866895f,    0.188938438822753f,
  -0.400894058284033f,    0.314370104391157f,     0.618580768947071f,
  0.830051263404639f,     -0.228700130023340f,    0.811855169643177f,
  0.0924092179787017f,    0.273652523319809f,     -0.0624274843235475f,
  -0.503696982048589f,    0.510545161203341f,     0.341823133345436f,
  -0.437486933663093f,    0.0134072800031224f,    0.613837993234983f,
  0.740945655313894f,     0.135311460882606f,     0.464832228842466f,
  -0.973962843371452f,    -0.519388256678232f,    0.631469277357519f,
  -0.936937468616713f,    0.208677911871604f,     -0.0946010975796272f,
  0.560587233611855f,     0.230925763372331f,     -0.637408482848184f,
  -0.679175194353885f,    -0.408696637706987f,    -0.0837464598184048f,
  -0.911070817707239f,    0.985815432104941f,     -0.208807972878988f,
  0.741966810464688f,     0.162772839973564f,     0.717702638881939f,
  0.490767958961575f,     -0.835565390813677f,    -0.878516167634055f,
  -0.956727838876563f,    -0.00772081382858891f,  0.355227897612178f,
  0.202889185809854f,     -0.431078767653467f,    0.106936101717808f,
  0.354494042302258f,     -0.619623833602791f,    0.193065593078352f,
  -0.105803087758606f,    0.151828327005194f,     -0.141094922099930f,
  0.847569902283069f,     -0.656683924792181f,    -0.880754505470701f,
  -0.421714047610595f,    0.681762288858050f,     0.633712681698887f,
  0.947060360650644f,     -0.959122611588459f,    -0.0690574969687099f,
  -0.805062392278087f,    0.226501754467861f,     -0.414732397455275f,
  0.242398867364043f,     -0.831838824773804f,    0.00787391802290793f,
  -0.860692119913991f,    -0.391321299589110f,    -0.0548681430681355f,
  -0.992920640472037f,    0.0975642331777702f,    0.894630836703243f,
  0.767919825689366f,     -0.260878774442215f,    0.407457430171103f,
  0.140688657702825f,     0.737494845272763f,     -0.650969054257040f,
  0.230613259000797f,     -0.0986876345046772f,   0.0996951163848971f,
  -0.679173062298700f,    -0.760174222364469f,    -0.613840714529317f,
  -0.692138390397415f,    -0.0919103790884603f,   0.0259548517830916f,
  0.463763807478796f,     -0.859327137970617f,    0.298600182982665f,
  -0.591236092977368f,    -0.994984881037264f,    -0.0533840054951049f,
  0.544979189292485f,     0.652482875230260f,     0.897548627394727f,
  -0.340241293753474f,    0.508237349558163f,     -0.611986702936889f,
  -0.399952468536369f,    -0.758494484998191f,    -0.148960755782999f,
  0.895231513826071f,     -0.870487943961511f,    -0.172763884748068f,
  -0.652702954266129f,    0.784450103085903f,     -0.428504279168614f,
  -0.347266234450861f,    -0.0897193897382391f,   0.760686883857503f,
  -0.0863659842493281f,   -0.453544362916610f,    0.713112885874267f,
  -0.529914378597266f,    -0.134507787695203f,    -0.590955798880753f,
  -0.372583442870916f,    0.646730663631020f,     -0.809515553972267f,
  0.0226873348847205f,    -0.209338539804651f,    -0.737170063193136f,
  0.365916689978321f,     0.658019395382111f,     0.733982378695990f,
  -0.579926149814113f,    0.973814182111372f,     0.933875763922095f,
  -0.985234946636757f,    -0.103124599698243f,    -0.798304574918884f,
  -0.119705341414667f,    0.205941898284561f,     0.111088288053652f,
  0.418598493379981f,     0.309112287901667f,     0.0865232803642195f,
  -0.281174085998345f,    -0.158426951248790f,    0.156672456990889f,
  0.608691108739118f,     -0.124654784531448f,    -0.372060827503666f,
  0.555750853569654f,     -0.481715370485256f,    0.411012047999522f,
  0.265636511301544f,     0.164466400718006f,     0.427292785417094,
  -0.407665783814271f,    0.0463239131527564f,    0.0109249300633605f,
  0.0949704798708169f,    0.223291931618591f,     0.708651599857453f,
  0.810927407452143f,     -0.298811874805995f,    0.347215272448441f,
  0.778225160999446f,     -0.981258755328673f,    -0.629231280170021f,
  -0.948786159268210f,    -0.0530522786747270f,   -0.665046033882002f,
  0.776993795678436f,     -0.604492154463805f,    -0.906360689482177f,
  0.543616910115371f,     -0.501547360013149f,    0.571784796850774f,
  0.868511495621889f,     0.783008382563488f,     0.571870376568081f,
  0.0471150346240308f,    0.402433510592678f,     0.661353159662286f,
  0.0253381317208246f,    0.720141243708461f,     -0.478805385943742f,
  0.989639021624774f,     0.538614599364854f,     -0.282810721919526f,
  0.888399971333007f,     0.118572990347886f,     0.564528382703688f,
  0.988296121763412f,     0.509638594649021f,     -0.797738059997026f,
  0.0363326380089621f,    0.978315833815278f,     -0.483368013204689f,
  0.879051054425480f,     0.632539830439323f,     0.722677742708361f,
  0.578919286433726f,     -0.250721628167261f,    0.534435049744896f,
  -0.0404568429105234f,   0.00805525426120179f,   0.841210870775473f,
  -0.731771544679396f,    0.713758914490801f,     0.830250762535296f,
  0.436563669872217f,     0.567024378980237f,     0.983941121609744f,
  -0.253548560865555f,    0.647105012325159f,     0.434994339049196f,
  0.130837710207442f,     -0.775136733344706f,    0.234917279141211f,
  -0.498429841761386f,    -0.273571256415041f,    0.247467425899991f,
  -0.970396693506149f,    0.975835855884816f,     -0.347896516486866f,
  -0.552856369180847f,    -0.887336234316568f,    -0.573271015958957f,
  0.910862901097874f,     -0.807236601077904f,    -0.523971593712952f,
  -0.263589563369279f,    0.591056276091253f,     -0.320168527954128f,
  0.726795865615521f,     -0.731502115921006f,    -0.942225519371229f,
  0.268573107637337f,     0.380348127119473f,     -0.284539943611895f,
  0.117478291379931f,     -0.817442486350524f,    0.0734705767013011f,
  -0.626880755668906f,    -0.873066996524459f,    -0.528675805715351f,
  0.490255491577847f,     0.398142666604162f,     -0.911320079669940f,
  -0.870350237514323f,    0.854587452657144f,     0.736349579728106f,
  0.948232845958681f,     -0.00126774478569258f,  0.905641169934000f,
  -0.965500575551565f,    0.0831330388550517f,    -0.892713267782484f,
  -0.277958019172831f,    0.312987842344813f,     0.484268977417485f,
  -0.365960524226328f,    0.177956605738091f,     0.913776767689874f,
  -0.897537691614058f,    0.473075982698961f,     0.913190042662185f,
  -0.00843862630950820f,  0.972679442298938f,     -0.856058592202917f,
  0.264007224067230f,     -0.138444823656136f,    -0.386195416368251f,
  -0.286657907928107f,    -0.231200657384828f,    0.917365701941188f,
  -0.271317547281263f,    -0.252691685075831f,    0.893742787021399f,
  0.512463051119608f,     0.979155111008605f,     -0.472272776864686f,
  0.238767541974988f,     -0.672234403865928f,    -0.846783135777377f,
  0.0877594573148737f,    0.493055606176910f,     -0.289012308379085f,
  0.416463895880697f,     -0.0795051375851281f,   -0.476692131327163f,
  -0.430471976159529f,    -0.701875030095239f,    0.724684336417516f,
  0.984802039066595f,     0.798285421773762f,     0.000509924988974175f,
  -0.0852199551444761f,   -0.709724122158260f,    -0.332735158942919f,
  -0.741119907407496f,    0.729608513555970f,     0.500578022862182f,
  0.520862987462710f,     0.565678561009731f,     -0.393741545311224f,
  -0.568866018100912f,    0.571654318026290f,     -0.817900961532165f,
  -0.793268461448962f,    0.614914392385859f,     0.763415306986536f,
  0.450074180772758f,     -0.737435729799608f,    0.841185794339245f,
  0.894276069286366f,     -0.276262284222369f,    -0.798623475612628f,
  -0.280994234105732f,    0.821175230597885f,     -0.474251640366966f,
  -0.190039801864015f,    0.0663032720971493f,    0.884162053156770f,
  -0.162023139878049f,    -0.963135153785511f,    -0.582213329804047f,
  -0.328536493809765f,    -0.938405687658462f,    -0.0171569611327957f,
  -0.727260847907578f,    0.419920927745257f,     -0.361592243835530f,
  0.476989471873569f,     -0.146161675185107f,    0.431817832405826f,
  -0.371528849369885f,    -0.940567978751516f,    0.165203770962029f,
  0.781321525273307f,     0.0585592625092357f,    0.573299596753579f,
  -0.378869924017182f,    0.523139576520889f,     0.385605607116478f,
  -0.235893429970747f,    0.285814921067909f,     -0.121941292771133f,
  0.621558611608942f,     -0.0860979132283732f,   -0.627097832687809f,
  -0.312083243705910f,    -0.494490796681559f,    -0.987187334387934f,
  -0.0804474888246625f,   0.496400656176795f,     -0.851811189314651f,
  -0.791398271297849f,    -0.868174317799275f,    -0.226794668997878f,
  -0.335339474552766f,    -0.276765924750817f,    -0.395876032147377f,
  -0.740529136126816f,    -0.167799472110453f,    0.593129248263724f,
  0.336783120133436f,     0.248892158925787f,     0.950120283075237f,
  -0.795216613504226f,    -0.574731116508357f,    -0.822689608026685f,
  0.973698546284335f,     0.125166556654624f,     0.588150318080073f,
  0.128654744345192f,     -0.219207714307262f,    -0.271053050307713f,
  0.124071241265810f,     -0.618209718015327f,    -0.766619799595349f,
  -0.478340220431165f,    -0.446873929629545f,    0.978019432749647f,
  -0.627041040766022f,    0.169323691066764f,     -0.714079827532216f,
  0.386101296128268f,     -0.360225804976135f,    -0.236797717782837f,
  -0.311635747131794f,    0.0482888201705840f,    -0.477302740867809f,
  -0.427349080854399f,    0.390352470816329f,     0.611790541936623f,
  -0.648292156214605f,    -0.345871618789073f,    0.509300603302844f,
  -0.0142202703124219f,   -0.570248077753979f,    -0.0629178211029751f,
  -0.737806048037047f,    0.497750084049821f,     -0.761650107803135f,
  -0.788756591098617f,    -0.994497286039420f,    -0.987344273533962f,
  0.657151987467984f,     -0.763708299084062f,    -0.0729359162118841f,
  0.0455275633022023f,    -0.101919187896584f,    0.457804242981095f,
  0.0117715388281796f,    -0.274125197027132f,    -0.949738804931191f,
  0.762108173886486f,     0.405150754308562f,     -0.733330375873553f,
  -0.712774896799572f,    -0.791947616412901f,    0.444023894424500f,
  0.00507562975249609f,   -0.900698136223538f,    -0.576055334977054f,
  -0.948895529956106f,    -0.832665060374124f,    -0.992753953473078f,
  -0.0674086978315183f,   0.569494111501383f,     -0.962269067721443f,
  -0.489700810475570f,    0.972626508328545f,     -0.777400448149780f,
  0.115588644128954f,     0.0730469703310024f,    0.523584488072518f,
  0.659055312807301f,     0.134198234373838f,     -0.797833055125151f,
  -0.167842823235145f,    -0.662347837139732f,    -0.537544370279756f,
  -0.622353549740796f,    -0.789789664448618f,    0.985300123665319f,
  0.862449845163424f,     0.973193986256980f,     0.148883268671144f,
  0.283619537155083f,     0.508503183151258f,     -0.246167305966866f,
  -0.259543094514413f,    -0.778029136807597f,    0.128978622849116f,
  -0.920978818238085f,    -0.116324837544276f,    -0.261472397833253f,
  0.772449038068069f,     -0.696754008784325f,    0.980778877985902f,
  -0.227636956328402f,    -0.472493776528032f,    -0.568519858000960f,
  -0.151689463117960f,    -0.102997587484899f,    0.464752146042376f,
  -0.839114793935065f,    -0.0325074343587592f,   -0.180618880765978f,
  0.0132253638432844f,    -0.646173464496730f,    0.821983901071593f,
  0.657453744559881f,     0.786315172070382f,     -0.438718096604728f,
  0.702691078885442f,     0.859957412428682f,     -0.505281395658564f,
  -0.236722160990303f,    -0.698465568366759f,    -0.746418979540090f,
  -0.218205126412646f,    -0.808715244840435f,    -0.949813739800491f,
  0.176975348790769f,     0.723960974918154f,     -0.139253733704369f,
  -0.387224393658603f,    -0.869945438924443f,    -0.396979039594941f,
  0.0256060022407458f,    -0.566074790002811f,    -0.161564565183606f,
  -0.736189868188370f,    -0.205593811665825f,    -0.628996407588712f,
  -0.0266462623004120f,   -0.344127255771429f,    -0.229003801178142f,
  -0.469786561635510f,    0.258249378153965f,     0.160442939158622f,
  0.0528817242116550f,    0.261960766261548f,     -0.571069557415276f,
  0.411333771884545f,     -0.145205354714326f,    0.249324532476397f,
  0.163889600722793f,     0.649915677347011f,     0.147077371087195f,
  -0.227104208942068f,    0.867390199578604f,     -0.0734153565896754f,
  0.0491208061060167f,    0.0360590744216485f,    0.181620126101180f,
  0.0567549454976457f,    -0.856976992549465f,    -0.242511192726339f,
  -0.624770508991394f,    -0.793161214564285f,    -0.251208532727981f,
  -0.833192309869275f,    0.368166434661069f,     0.939730260791580f,
  0.305796202211942f,     -0.598830491282818f,    -0.0575368190467946f,
  0.371329658849021f,     -0.227872714677810f,    0.707539568196379f,
  0.795186297468385f,     0.475847791658551f,     0.829361555893632f,
  0.405386540930889f,     0.213282954068900f,     0.767339023510319f,
  0.525055513018554f,     0.259437496637378f,     -0.524342591286100f,
  -0.731515526086696f,    -0.233118783725590f,    0.237972339628935f,
  -0.933985285078109f,    0.537013420173496f,     0.498819465200784f,
  -0.407713459607516f,    0.382821417923595f,     -0.416894700661466f,
  0.0787266904103943f,    -0.0627973593192392f,   -0.320105342653426f,
  -0.844066834407447f,    0.138221622417319f,     -0.676665423871596f,
  -0.961043785105959f,    0.832268610130385f,     -0.905530890441773f,
  -0.114191325652611f,    -0.376697124207843f,    0.390323137798417f,
  0.953143142925101f,     0.983427991280007f,     -0.0895687386326503f,
  -0.681543125061097f,    0.677131540142176f,     -0.867715848764628f,
  -0.812718786220309f,    -0.212509939486840f,    -0.990002327123638f,
  -0.0682855560011961f,   0.129310729289606f,     -0.623746296335073f,
  -0.285580241032587f,    0.235626081900812f,     -0.611973228709249f,
  0.539189737955466f,     0.970058678533189f,     0.901944358898624f,
  0.168094826408153f,     -0.666711281252260f,    0.965612752173968f,
  0.651034558458719f,     0.687501917067508f,     0.758614314567106f,
  -0.839396045781239f,    -0.552775028233564f,    -0.528941743867256f,
  0.174761156721889f,     0.243585712774679f,     0.588913151268911f,
  -0.306898192880627f,    0.921540023069231f,     -0.0223654942298541f,
  -0.102408576957649f,    0.612577852207921f,     0.835809058447089f,
  -0.437118459197839f,    0.455316033239981f,     0.311435507416257f,
  -0.648992336007256f,    0.346823844785409f,     -0.632080213667648f,
  -0.599678627679202f,    -0.653822991854328f,    0.484305292443427f,
  0.782046295685087f,     0.960987598814982f,     0.627169162605570f,
  0.948092598306120f,     -0.185268381817018f,    0.602489977060513f,
  -0.885827813790617f,    -0.00179203147433582f,  -0.175476447614991f,
  0.0461282236560925f,    -0.898013889602944f,    0.256310837914276f,
  -0.733422110056865f,    -0.740091677658095f,    0.966724540497493f,
  0.328056986822768f,     -0.267854591449557f,    0.670545831663244f,
  -0.356204313297688f,    0.0729865206358908f,    -0.594530723723669f,
  0.519965324048968f,     0.0632129606097647f,    -0.878434885663544f,
  -0.497945943395010f,    0.0151854050905818f,    -0.218036856012343f,
  0.547721213710874f,     -0.0915514918588898f,   -0.279344098401951f,
  -0.228654882218650f,    0.100432155997130f,     0.802024600677294f,
  0.175832345686877f,     0.0551231013299744f,    0.938247319394824f,
  0.639298571360036f,     -0.291461603371678f,    -0.853503115314794f,
  -0.604829242631156f,    0.0291571486740745f,    -0.932575328418390f,
  -0.621235088415116f,    0.403040314052094f,     -0.809695618266849f,
  0.966605888732736f,     -0.199254401023053,     -0.540808222970056f,
  -0.0141840769249790f,   0.114579158224093f,     0.466889318471371f,
  -0.145415211797766f,    -0.846707387707480f,    -0.881237200733915f,
  -0.410798723199712f,    -0.637697860299854f,    -0.196366036081372f,
  0.193267531425712f,     -0.258591200841940f,    -0.173003722066551f,
  0.478121376006132f,     0.953819951501542f,     0.969916001975448f,
  0.131515861287576f,     -0.499829658784781f,    0.320952777516193f,
  -0.226980682212371f,    0.766886115655233f,     0.647310434110803f,
  -0.772594685974992f,    0.772645949480187f,     -0.936357605801364f,
  -0.671842916281206f,    -0.595127074295355f,    0.335132581825520f,
  0.648964430112689f,     -0.793376819398441f,    -0.963663232647360f,
  0.914308824314478f,     -0.397663128784982f,    0.803240040231588f,
  -0.291039120047626f,    -0.339918835846510f,    -0.208620988780609f,
  0.278177231697424f,     -0.833157746552451f,    0.260554706029473f,
  -0.580537744139231f,    0.918561093477862f,     0.641368468308093f,
  0.827379039283645f,     -0.412231303854834f,    -0.518315486749742f,
  0.423356687848085f,     0.0777277584993787f,    0.394127392657178f,
  0.609705410002715f,     0.264669032561337f,     -0.460555696512027f,
  -0.0858908123066196f,   -0.281781559603429f,    -0.179777723960362f,
  -0.00449990348855067f,  0.803703377739133f,     -0.155074032314596f,
  -0.00206139428833696f,  0.0661730930565525f,    -0.737509215752079f,
  0.620182143819587f,     0.114750705414661f,     0.545663051433958f,
  0.661601724477194f,     -0.592280382351976f,    0.609240020031149f,
  -0.968781019917808f,    -0.668068368389875f,    0.206915551463500f,
  0.0951453192552747f,    0.268580107578401f,     -0.0450052302342363f,
  -0.933589842483940f,    0.236570492858402f,     0.0688734168318912f,
  0.930163232697303f,     0.435953476823146f,     0.533759385687075f,
  0.368282038662015f,     -0.602312961473778f,    0.709516631712345f,
  -0.168303926671961f,    0.130670870119294f,     -0.657736111745007f,
  0.115028598388756f,     0.173728026281032f,     -0.681671363429886f,
  -0.538786035950873f,    0.481457671665448f,     0.0136795278434168f,
  -0.570065342244344f,    0.188187050857249f,     -0.352869308173680f,
  -0.979175308628854f,    0.223702879460018f,     0.994220466087713f,
  -0.147795166105729f,    0.218427535879435f,     -0.120050826084179f,
  -0.0124939247430063f,   -0.645134875027126f,    -0.503122688484778f,
  0.534123007328982f,     0.619710972635444f,     -0.234248243706177f,
  0.987144458053815f,     0.261284702576427f,     0.851827092094236f,
  0.750019654249059f,     -0.926154630610335f,    0.449356103243440f,
  0.783011320523296f,     -0.459228158107270f,    -0.228877816937867f,
  0.271108937592868f,     -0.676085611673506f,    0.783114428240160f,
  0.636093784021493f,     -0.754110314308629f,    -0.546386104880684f,
  0.0385811136139234f,    -0.768951137117397f,    -0.644624743947807f,
  0.00890157035391148f,   -0.0792572116273387f,   -0.989980668770044f,
  0.603057533157075f,     0.280835727469123f,     -0.634716709420524f,
  -0.712669415138995f,    -0.424129916157595f,    -0.436923748487354f,
  0.467366013559791f,     0.907740481011987f,     0.788617065944311f,
  -0.152237692069130f,    -0.963044404518533f,    0.907393322909416f,
  0.806867676446313f,     0.699270310021791f,     0.107867603776547f,
  0.127360747415417f,     -0.502645789696788f,    -0.511744166872327f,
  -0.121672719343072f,    -0.596527146770249f,    0.410180172377510f,
  -0.852889849908704f,    0.278972213674154f,     0.0260156356783650f,
  0.997558501949683f,     -0.499245840292893f,    -0.451169267624132f,
  -0.881643497362337f,    0.986957209874262f,     -0.129608292321380f,
  0.935829016346258f,     -0.649021465281634f,    0.550436689069794f,
  0.278888743082679f,     0.0137769346664500f,    -0.660666060213522f,
  -0.416709136728042f,    -0.302903068397225f,    0.180657445835459f,
  -0.908195955986293f,    0.280056533234627f,     -0.660025789034158f,
  -0.798207438952561f,    0.901575224780405f,     -0.608702932295102f,
  0.318860199910414f,     0.874005722023406f,     -0.0816057579181704f,
  0.981671341873017f,     -0.339234700161323f,    0.559717959858931f,
  0.390363525109105f,     -0.309384476087470f,    0.956563156784297f,
  -0.623734354817613f,    -0.196627375289105f,    -0.702076014509088f,
  0.293098766889643f,     -0.617152224560368f,    0.859117491438645f,
  0.661015739867647f,     0.0747554166353739f,    -0.282417009682732f,
  -0.667461537762524f,    -0.451029960388404f,    -0.464518668674360f,
  0.591389440503293f,     0.552648871601186f,     -0.242406315814918f,
  0.147876771864331f,     -0.00605730052917419f,  -0.850648363553678f,
  -0.659957159486993f,    -0.165475362851332f,    0.204150315434812f,
  -0.665767311591476f,    -0.716154682563576f,    0.417487456932076f,
  0.448184990956287f,     0.733843802413198f,     -0.170228277851921f,
  -0.346809954182150f,    0.956058632188011f,     0.0315623945930987f,
  0.509027121691627f,     -0.147826185909834f,    0.717423768198044f,
  -0.153258078639530f,    -0.586190749016474f,    0.122228033051868f,
  -0.884999045468193f,    -0.364729711773548f,    0.0869976154696972f,
  -0.793532199218799f,    0.533748273468951f,     -0.852754376244435f,
  0.294752047699830f,     0.136764278163013f,     0.838074791168389f,
  0.795224598541123f,     -0.778005568697498f,    -0.260924769562304f,
  -0.303759147861313f,    0.273726011325558f,     0.530476779331216f,
  0.0866801234357086f,    0.0677702376031544f,    0.724353404182035f,
  -0.974710312543683f,    0.791838170482991f,     0.247768259921660f,
  0.979431048271259f,     -0.386992541899814f,    0.0640038231299192f,
  -0.00457726816166693f,  0.371455553726539f,     0.647649995487707f,
  0.268304945808406f,     -0.320428608173924f,    0.0927511620029309f,
  0.256010036486838f,     0.740396212690346f,     -0.656873241472848f,
  0.823534292439413f,     -0.820380362458844f,    -0.453300307443023f,
  0.784238355222248f,     0.912791840124321f,     0.0999478035440859f,
  -0.212620916988855f,    0.0170290625008669f,    -0.589062380565879f,
  -0.171833624145497f,    -0.524918122866141f,    0.961528292650892f,
  0.101262818636430f,     0.941455114569308f,     -0.967226220138929f,
  0.616781547648562f,     -0.913823148383971f,    0.274508821885917f,
  0.924653374107756f,     -0.866302908989783f,    0.227541907715857f,
  0.0907574361370582f,    -0.127499097943315f,    -0.942071371521895f,
  -0.119419163649152f,    0.674284088819523f,     0.881328505929745f,
  0.246290207551702f,     0.0547607254148590f,    -0.462882918359077f,
  0.888969728230585f,     0.666583509571921f,     0.238417203582380f,
  -0.279842248122727f,    0.855260336845903f,     0.314306869401155f,
  -0.188654877893078f,    -0.609304918228150f,    0.169453885325888f,
  0.265617874907016f,     -0.943423537926184f,    0.493118676869450f,
  -0.386147750024858f,    0.0103920154342951f,    0.753677832518483f,
  0.363353012331066f,     -0.286620106520429f,    -0.623332994906295f,
  0.183966714365642f,     -0.124278942882867f,    -0.687889346448110f,
  -0.509002319646341f,    -0.705769785650865f,    0.600586126467172f,
  0.814989294939922f,     0.198959025256652f,     0.477897007911356f,
  0.757957814363899f,     0.617755921094230f,     -0.353589871947529f,
  0.419688673189503f,     -0.860584865805600f,    -0.0232779635193843f,
  -0.789951030889387f,    -0.893196700185750f,    0.610996462535201f,
  0.847373590985131f,     -0.989553358501822f,    -0.367651771428081f,
  0.741563712056747f,     -0.923595352848971f,    -0.580174215739367f,
  0.577092000574232f,     -0.910872910110270f,    -0.907499077314190f,
  0.692372247654077f,     0.810694134592084f,     -0.608596332548047f,
  0.761254615051625f,     0.0546240611947364f,    -0.393956427117691f,
  -0.116127831535139f,    -0.0352014590913388f,   0.374742194768889f,
  -0.927344099730091f,    0.939301337232488f,     -0.969831716293845f,
  -0.0489333404770240f,   -0.586719398908953f,    0.0235541378462407f,
  0.388882981728285f,     -0.0728483242295113f,   0.418280445244943f,
  -0.574289337805456f,    -0.779962057565259f,    -0.835190719754123f,
  0.918717316922657f,     -0.765889988109173f,    -0.935310664146932f,
  -0.0750906135370848f,   -0.256246546197534f,    0.693865929543926f,
  0.592800255527084f,     0.836743344551035f,     -0.801953470827580f,
  0.0595524153568945f,    0.158376549012192f,     -0.429364776412726f,
  -0.450531184162532f,    -0.169317185285268f,    0.420344570579195f,
  -0.902838087574441f,    -0.654676904337469f,    0.941802178622893f,
  -0.411034608875500f,    -0.455381371659872f,    0.582371240315256f,
  -0.276150504466756f,    0.164276403376185f,     -0.960238817086774f,
  0.590055303394028f,     -0.995185688656226f,    -0.285809748360467f,
  -0.792066171752882f,    -0.456123303649101f,    -0.864169187700384f,
  0.798745251308383f,     -0.517673464079948f,    0.523086536900369f,
  0.398784615211052f,     0.908677185333852f,     -0.434846969584770f,
  -0.277024535706464f,    0.575800013122065f,     -0.0423952171673019f,
  -0.327530749916683f,    -0.401220909875065f,    -0.232577533032385f,
  0.577630268254944f,     -0.733290201484409f,    -0.297499739456838f,
  0.166541885572822f,     -0.646828619904039f,    0.0312662656272755f,
  0.754145050600965f,     -0.908499825108811f,    0.315379190361296f,
  0.366242661082351f,     0.867903806940678f,     -0.613391940567782f,
  0.00760147209048068f,   0.953424134034927f,     -0.812551125910811f,
  0.734998935207065f,     0.781720854678504f,     -0.653974423413561f,
  0.612587888218526f,     -0.297359914095386f,    -0.409559158758694f,
  -0.143962230212734f,    -0.814888102841114f,    0.359131810502055f,
  0.356924557741016f,     -0.872415989401612f,    0.716849887848809f,
  -0.374916928585818f,    -0.0702264435280595f,   0.329843438660215f,
  0.0956097573088677f,    -0.937528128860310f,    -0.322293489817529f,
  0.781444964993177f,     -0.810141738751828f,    -0.150295079242497f,
  0.846909181293597f,     -0.128124277517711f,    -0.752611695817661f,
  0.839996835828451f,     -0.0705685941510277f,   0.000366462394740585f,
  0.0788016995849923f,    -0.246053200655556f,    -0.156645099915028f,
  -0.460752333796863f,    0.622021359864204f,     0.722871957583123f,
  -0.257873353238499f,    -0.309810184480446f,    0.765248644407833f,
  -0.553316047243663f,    -0.612742789838491f,    0.354017349601509f,
  0.923293368553697f,     0.630695912377860f,     -0.148750121613537f,
  -0.821801680234240f,    0.368247966228196f,     0.405416044101496f,
  -0.803232509711354f,    -0.429778551911399f,    -0.723837414527446f,
  -0.963925147452133f,    0.190882872226757f,     0.477008077263598f,
  -0.661282403679070f,    0.271643442525556f,     -0.915994079618801f,
  0.196564556546175f,     0.378359035245796f,     0.584016730657668f,
  -0.0377864332655202f,   -0.327376192853106f,    0.850744189707984f,
  0.799571679043808f,     -0.111126908452029f,    0.525587242291601f,
  -0.404486180733535f,    -0.134496922397279f,    0.0890128096708100f,
  -0.815560643303157f,    -0.920166023598312f,    -0.360079578314899f,
  -0.556238898466371f,    -0.220978103133838f,    -0.571530268052405f,
  0.573332217175226f,     -0.133862258696460f,    -0.982130330352248f,
  -0.352538465285082f,    0.318683937697894f,     -0.790927430842686f,
  0.691168535237102f,     0.806014327242002f,     -0.981639450008060f,
  0.407200095027265f,     0.918249921845949f,     0.776880149695420f,
  -0.437773083955269f,    -0.385117533333437f,    0.0115152415796460f,
  0.687224538003991f,     0.992524870612626f,     0.471003324792228f,
  -0.873541777412034f,    -0.560923118634380f,    -0.726151823613842f,
  -0.538941951730010f,    0.772057551475325f,     0.858490725829641f,
  -0.168849338472479f};
# 100 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/fixed-point.h"
fxp_t wrap(fxp_t kX, fxp_t kLowerBound, fxp_t kUpperBound)
{
  int32_t range_size = kUpperBound - kLowerBound + 1;
  if(kX < kLowerBound)
  {
    kX += range_size * ((kLowerBound - kX) / range_size + 1);
  }
  return kLowerBound + (kX - kLowerBound) % range_size;
}

fxp_t fxp_get_int_part(fxp_t in)
{
  return ((in < 0) ? -((-in) & _fxp_imask) : in & _fxp_imask);
}

fxp_t fxp_get_frac_part(fxp_t in)
{
  return ((in < 0) ? -((-in) & _fxp_fmask) : in & _fxp_fmask);
}

float fxp_to_float(fxp_t fxp);

fxp_t fxp_quantize(fxp_t aquant)
{
  if(overflow_mode == 2)
  {
    if(aquant < _fxp_min)
    {
      return _fxp_min;
    }
    else if(aquant > _fxp_max)
    {
      return _fxp_max;
    }
  }
  else if(overflow_mode == 3)
  {
    if(aquant < _fxp_min || aquant > _fxp_max)
    {
      return wrap(aquant, _fxp_min, _fxp_max);
    }
  }
  return (fxp_t)aquant;
}

void fxp_verify_overflow(fxp_t value)
{
  fxp_quantize(value);
  __DSVERIFIER_assert(value <= _fxp_max && value >= _fxp_min);
}

void fxp_verify_overflow_node(fxp_t value)
{
  if(1 == 2)
  {
    __DSVERIFIER_assert(value <= _fxp_max && value >= _fxp_min);
  }
}

void fxp_verify_overflow_array(fxp_t array[], int n)
{
  int i = 0;
  for(i = 0; i < n; i++)
  {
    fxp_verify_overflow(array[i]);
  }
}

fxp_t fxp_int_to_fxp(int in)
{
  fxp_t lin;
  lin = (fxp_t)in * _fxp_one;
  return lin;
}

int fxp_to_int(fxp_t fxp)
{
  if(fxp >= 0)
  {
    fxp += _fxp_half;
  }
  else
  {
    fxp -= _fxp_half;
  }
  fxp >>= impl.frac_bits;
  return (int)fxp;
}

fxp_t fxp_float_to_fxp(float f)
{
  fxp_t tmp;
  double ftemp;
  ftemp = f * scale_factor[impl.frac_bits];
  if(f >= 0)
  {
    tmp = (fxp_t)(ftemp + 0.5);
  }
  else
  {
    tmp = (fxp_t)(ftemp - 0.5);
  }
  return tmp;
}

fxp_t fxp_double_to_fxp(double value)
{
  fxp_t tmp;
  double ftemp = value * scale_factor[impl.frac_bits];
  if(rounding_mode == 0)
  {
    if(value >= 0)
    {
      tmp = (fxp_t)(ftemp + 0.5);
    }
    else
    {
      tmp = (fxp_t)(ftemp - 0.5);
    }
  }
  else if(rounding_mode == 1)
  {
    tmp = (fxp_t)ftemp;
    double residue = ftemp - tmp;
    if((value < 0) && (residue != 0))
    {
      ftemp = ftemp - 1;
      tmp = (fxp_t)ftemp;
    }
  }
  else if(rounding_mode == 0)
  {
    tmp = (fxp_t)ftemp;
  }
  return tmp;
}

void fxp_float_to_fxp_array(float f[], fxp_t r[], int N)
{
  int i;
  for(i = 0; i < N; ++i)
  {
    r[i] = fxp_float_to_fxp(f[i]);
  }
}

void fxp_double_to_fxp_array(double f[], fxp_t r[], int N)
{
  int i;
  for(i = 0; i < N; ++i)
  {
    r[i] = fxp_double_to_fxp(f[i]);
  }
}
# 271 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/fixed-point.h"
float fxp_to_float(fxp_t fxp)
{
  float f;
  int f_int = (int)fxp;
  f = f_int * scale_factor_inv[impl.frac_bits];
  return f;
}

double fxp_to_double(fxp_t fxp)
{
  double f;
  int f_int = (int)fxp;
  f = f_int * scale_factor_inv[impl.frac_bits];
  return f;
}

void fxp_to_float_array(float f[], fxp_t r[], int N)
{
  int i;
  for(i = 0; i < N; ++i)
  {
    f[i] = fxp_to_float(r[i]);
  }
}

void fxp_to_double_array(double f[], fxp_t r[], int N)
{
  int i;
  for(i = 0; i < N; ++i)
  {
    f[i] = fxp_to_double(r[i]);
  }
}

fxp_t fxp_abs(fxp_t a)
{
  fxp_t tmp;
  tmp = ((a < 0) ? -(fxp_t)(a) : a);
  fxp_verify_overflow_node(tmp);
  tmp = fxp_quantize(tmp);
  return tmp;
}

fxp_t fxp_add(fxp_t aadd, fxp_t badd)
{
  fxp_t tmpadd;
  tmpadd = ((fxp_t)(aadd) + (fxp_t)(badd));
  fxp_verify_overflow_node(tmpadd);
  tmpadd = fxp_quantize(tmpadd);
  return tmpadd;
}

fxp_t fxp_sub(fxp_t asub, fxp_t bsub)
{
  fxp_t tmpsub;
  tmpsub = (fxp_t)((fxp_t)(asub) - (fxp_t)(bsub));
  fxp_verify_overflow_node(tmpsub);
  tmpsub = fxp_quantize(tmpsub);
  return tmpsub;
}

fxp_t fxp_mult(fxp_t amult, fxp_t bmult)
{
  fxp_t tmpmult, tmpmultprec;
  tmpmult = (fxp_t)((fxp_t)(amult) * (fxp_t)(bmult));
  if(tmpmult >= 0)
  {
    tmpmultprec = (tmpmult + ((tmpmult & 1 << (impl.frac_bits - 1)) << 1)) >>
                  impl.frac_bits;
  }
  else
  {
    tmpmultprec =
      -(((-tmpmult) + (((-tmpmult) & 1 << (impl.frac_bits - 1)) << 1)) >>
        impl.frac_bits);
  }
  fxp_verify_overflow_node(tmpmultprec);
  tmpmultprec = fxp_quantize(tmpmultprec);
  return tmpmultprec;
}
# 372 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/fixed-point.h"
fxp_t fxp_div(fxp_t a, fxp_t b)
{
  fxp_t tmpdiv = ((a << impl.frac_bits) / b);
  fxp_verify_overflow_node(tmpdiv);
  tmpdiv = fxp_quantize(tmpdiv);
  return tmpdiv;
}

fxp_t fxp_neg(fxp_t aneg)
{
  fxp_t tmpneg;
  tmpneg = -(fxp_t)(aneg);
  fxp_verify_overflow_node(tmpneg);
  tmpneg = fxp_quantize(tmpneg);
  return tmpneg;
}
# 399 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/fixed-point.h"
fxp_t fxp_sign(fxp_t a)
{
  return ((a == 0) ? 0 : ((a < 0) ? _fxp_minus_one : _fxp_one));
}

fxp_t fxp_shrl(fxp_t in, int shift)
{
  return (fxp_t)(((unsigned int)in) >> shift);
}

fxp_t fxp_square(fxp_t a)
{
  return fxp_mult(a, a);
}

void fxp_print_int(fxp_t a)
{
  printf("\n%i", (int32_t)a);
}

void fxp_print_float(fxp_t a)
{
  printf("\n%f", fxp_to_float(a));
}

void fxp_print_float_array(fxp_t a[], int N)
{
  int i;
  for(i = 0; i < N; ++i)
  {
    printf("\n%f", fxp_to_float(a[i]));
  }
}

void print_fxp_array_elements(char *name, fxp_t *v, int n)
{
  printf("%s = {", name);
  int i;
  for(i = 0; i < n; i++)
  {
    printf(" %jd ", v[i]);
  }
  printf("}\n");
}
# 21 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/util.h" 1
# 22 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/util.h"
void initialize_array(double v[], int n)
{
  int i;
  for(i = 0; i < n; i++)
  {
    v[i] = 0;
  }
}

void revert_array(double v[], double out[], int n)
{
  initialize_array(out, n);
  int i;
  for(i = 0; i < n; i++)
  {
    out[i] = v[n - i - 1];
  }
}

double internal_pow(double a, double b)
{
  int i;
  double acc = 1;
  for(i = 0; i < b; i++)
  {
    acc = acc * a;
  }
  return acc;
}

double internal_abs(double a)
{
  return a < 0 ? -a : a;
}

int fatorial(int n)
{
  return n == 0 ? 1 : n * fatorial(n - 1);
}

int check_stability(double a[], int n)
{
  int lines = 2 * n - 1;
  int columns = n;
  double m[lines][n];
  int i, j;

  double current_stability[n];
  for(i = 0; i < n; i++)
  {
    current_stability[i] = a[i];
  }

  double sum = 0;
  for(i = 0; i < n; i++)
  {
    sum += a[i];
  }
  if(sum <= 0)
  {
    printf("[DEBUG] the first constraint of Jury criteria failed: (F(1) > 0)");
    return 0;
  }

  sum = 0;
  for(i = 0; i < n; i++)
  {
    sum += a[i] * internal_pow(-1, n - 1 - i);
  }
  sum = sum * internal_pow(-1, n - 1);
  if(sum <= 0)
  {
    printf(
      "[DEBUG] the second constraint of Jury criteria failed: (F(-1)*(-1)^n > "
      "0)");
    return 0;
  }

  if(internal_abs(a[n - 1]) > a[0])
  {
    printf(
      "[DEBUG] the third constraint of Jury criteria failed: (abs(a0) < "
      "a_{n}*z^{n})");
    return 0;
  }

  for(i = 0; i < lines; i++)
  {
    for(j = 0; j < columns; j++)
    {
      m[i][j] = 0;
    }
  }
  for(i = 0; i < lines; i++)
  {
    for(j = 0; j < columns; j++)
    {
      if(i == 0)
      {
        m[i][j] = a[j];
        continue;
      }
      if(i % 2 != 0)
      {
        int x;
        for(x = 0; x < columns; x++)
        {
          m[i][x] = m[i - 1][columns - x - 1];
        }
        columns = columns - 1;
        j = columns;
      }
      else
      {
        m[i][j] = m[i - 2][j] - (m[i - 2][columns] / m[i - 2][0]) * m[i - 1][j];
      }
    }
  }
  int first_is_positive = m[0][0] >= 0 ? 1 : 0;
  for(i = 0; i < lines; i++)
  {
    if(i % 2 == 0)
    {
      int line_is_positive = m[i][0] >= 0 ? 1 : 0;
      if(first_is_positive != line_is_positive)
      {
        return 0;
      }
      continue;
    }
  }
  return 1;
}

void poly_sum(double a[], int Na, double b[], int Nb, double ans[], int Nans)
{
  int i;
  Nans = Na > Nb ? Na : Nb;

  for(i = 0; i < Nans; i++)
  {
    if(Na > Nb)
    {
      ans[i] = a[i];
      if(i > Na - Nb - 1)
      {
        ans[i] = ans[i] + b[i - Na + Nb];
      }
    }
    else
    {
      ans[i] = b[i];
      if(i > Nb - Na - 1)
      {
        ans[i] = ans[i] + a[i - Nb + Na];
      }
    }
  }
}

void poly_mult(double a[], int Na, double b[], int Nb, double ans[], int Nans)
{
  int i;
  int j;
  int k;
  Nans = Na + Nb - 1;

  for(i = 0; i < Na; i++)
  {
    for(j = 0; j < Nb; j++)
    {
      k = Na + Nb - i - j - 2;
      ans[k] = 0;
    }
  }

  for(i = 0; i < Na; i++)
  {
    for(j = 0; j < Nb; j++)
    {
      k = Na + Nb - i - j - 2;
      ans[k] = ans[k] + a[Na - i - 1] * b[Nb - j - 1];
    }
  }
}

void double_check_oscillations(double *y, int y_size)
{
  __DSVERIFIER_assume(y[0] != y[y_size - 1]);
  int window_timer = 0;
  int window_count = 0;
  int i, j;
  for(i = 2; i < y_size; i++)
  {
    int window_size = i;
    for(j = 0; j < y_size; j++)
    {
      if(window_timer > window_size)
      {
        window_timer = 0;
        window_count = 0;
      }

      int window_index = j + window_size;
      if(window_index < y_size)
      {
        if(y[j] == y[window_index])
        {
          window_count++;

          ((!(window_count == window_size))
             ? (void)0
             : _assert(
                 "!(window_count == window_size)",
                 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/util.h",
                 207));
        }
      }
      else
      {
        break;
      }
      window_timer++;
    }
  }
}

void double_check_limit_cycle(double *y, int y_size)
{
  double reference = y[y_size - 1];
  int idx = 0;
  int window_size = 1;

  for(idx = (y_size - 2); idx >= 0; idx--)
  {
    if(y[idx] != reference)
    {
      window_size++;
    }
    else
    {
      break;
    }
  }

  __DSVERIFIER_assume(window_size != y_size && window_size != 1);
  printf("window_size %d\n", window_size);
  int desired_elements = 2 * window_size;
  int found_elements = 0;

  for(idx = (y_size - 1); idx >= 0; idx--)
  {
    if(idx > (y_size - window_size - 1))
    {
      printf("%.0f == %.0f\n", y[idx], y[idx - window_size]);
      int cmp_idx = idx - window_size;
      if((cmp_idx > 0) && (y[idx] == y[idx - window_size]))
      {
        found_elements = found_elements + 2;
      }
      else
      {
        break;
      }
    }
  }
  printf("desired_elements %d\n", desired_elements);
  printf("found_elements %d\n", found_elements);
  __DSVERIFIER_assert(desired_elements != found_elements);
}

void double_check_persistent_limit_cycle(double *y, int y_size)
{
  int idy = 0;
  int count_same = 0;
  int window_size = 0;
  double reference = y[0];

  for(idy = 0; idy < y_size; idy++)
  {
    if(y[idy] != reference)
    {
      window_size++;
    }
    else if(window_size != 0)
    {
      break;
    }
    else
    {
      count_same++;
    }
  }
  window_size += count_same;

  __DSVERIFIER_assume(window_size > 1 && window_size <= y_size / 2);

  double lco_elements[window_size];
  for(idy = 0; idy < y_size; idy++)
  {
    if(idy < window_size)
    {
      lco_elements[idy] = y[idy];
    }
  }

  idy = 0;
  int lco_idy = 0;
  _Bool is_persistent = 0;
  while(idy < y_size)
  {
    if(y[idy++] == lco_elements[lco_idy++])
    {
      is_persistent = 1;
    }
    else
    {
      is_persistent = 0;
      break;
    }

    if(lco_idy == window_size)
    {
      lco_idy = 0;
    }
  }
  __DSVERIFIER_assert(is_persistent == 0);
}

void print_array_elements(char *name, double *v, int n)
{
  printf("%s = {", name);
  int i;
  for(i = 0; i < n; i++)
  {
    printf(" %.32f ", v[i]);
  }
  printf("}\n");
}

void double_add_matrix(
  unsigned int lines,
  unsigned int columns,
  double m1[4][4],
  double m2[4][4],
  double result[4][4])
{
  unsigned int i, j;
  for(i = 0; i < lines; i++)
  {
    for(j = 0; j < columns; j++)
    {
      result[i][j] = m1[i][j] + m2[i][j];
    }
  }
}

void double_sub_matrix(
  unsigned int lines,
  unsigned int columns,
  double m1[4][4],
  double m2[4][4],
  double result[4][4])
{
  unsigned int i, j;
  for(i = 0; i < lines; i++)
  {
    for(j = 0; j < columns; j++)
    {
      result[i][j] = m1[i][j] - m2[i][j];
    }
  }
}

void double_matrix_multiplication(
  unsigned int i1,
  unsigned int j1,
  unsigned int i2,
  unsigned int j2,
  double m1[4][4],
  double m2[4][4],
  double m3[4][4])
{
  unsigned int i, j, k;
  if(j1 == i2)
  {
    for(i = 0; i < i1; i++)
    {
      for(j = 0; j < j2; j++)
      {
        m3[i][j] = 0;
      }
    }

    for(i = 0; i < i1; i++)
    {
      for(j = 0; j < j2; j++)
      {
        for(k = 0; k < j1; k++)
        {
          double mult = (m1[i][k] * m2[k][j]);

          m3[i][j] = m3[i][j] + (m1[i][k] * m2[k][j]);
        }
      }
    }
  }
  else
  {
    printf("\nError! Operation invalid, please enter with valid matrices.\n");
  }
}

void fxp_matrix_multiplication(
  unsigned int i1,
  unsigned int j1,
  unsigned int i2,
  unsigned int j2,
  fxp_t m1[4][4],
  fxp_t m2[4][4],
  fxp_t m3[4][4])
{
  unsigned int i, j, k;
  if(j1 == i2)
  {
    for(i = 0; i < i1; i++)
    {
      for(j = 0; j < j2; j++)
      {
        m3[i][j] = 0;
      }
    }

    for(i = 0; i < i1; i++)
    {
      for(j = 0; j < j2; j++)
      {
        for(k = 0; k < j1; k++)
        {
          m3[i][j] = fxp_add(m3[i][j], fxp_mult(m1[i][k], m2[k][j]));
        }
      }
    }
  }
  else
  {
    printf("\nError! Operation invalid, please enter with valid matrices.\n");
  }
}

void fxp_exp_matrix(
  unsigned int lines,
  unsigned int columns,
  fxp_t m1[4][4],
  unsigned int expNumber,
  fxp_t result[4][4])
{
  unsigned int i, j, l, k;
  fxp_t m2[4][4];

  if(expNumber == 0)
  {
    for(i = 0; i < lines; i++)
    {
      for(j = 0; j < columns; j++)
      {
        if(i == j)
        {
          result[i][j] = fxp_double_to_fxp(1.0);
        }
        else
        {
          result[i][j] = 0.0;
        }
      }
    }
    return;
  }

  for(i = 0; i < lines; i++)
    for(j = 0; j < columns; j++)
      result[i][j] = m1[i][j];

  if(expNumber == 1)
  {
    return;
  }
  for(l = 1; l < expNumber; l++)
  {
    for(i = 0; i < lines; i++)
      for(j = 0; j < columns; j++)
        m2[i][j] = result[i][j];
    for(i = 0; i < lines; i++)
      for(j = 0; j < columns; j++)
        result[i][j] = 0;
    for(i = 0; i < lines; i++)
    {
      for(j = 0; j < columns; j++)
      {
        for(k = 0; k < columns; k++)
        {
          result[i][j] = fxp_add(result[i][j], fxp_mult(m2[i][k], m1[k][j]));
        }
      }
    }
  }
}

void double_exp_matrix(
  unsigned int lines,
  unsigned int columns,
  double m1[4][4],
  unsigned int expNumber,
  double result[4][4])
{
  unsigned int i, j, k, l;
  double m2[4][4];

  if(expNumber == 0)
  {
    for(i = 0; i < lines; i++)
    {
      for(j = 0; j < columns; j++)
      {
        if(i == j)
        {
          result[i][j] = 1.0;
        }
        else
        {
          result[i][j] = 0.0;
        }
      }
    }
    return;
  }

  for(i = 0; i < lines; i++)
    for(j = 0; j < columns; j++)
      result[i][j] = m1[i][j];

  if(expNumber == 1)
  {
    return;
  }
  for(l = 1; l < expNumber; l++)
  {
    for(i = 0; i < lines; i++)
      for(j = 0; j < columns; j++)
        m2[i][j] = result[i][j];
    for(i = 0; i < lines; i++)
      for(j = 0; j < columns; j++)
        result[i][j] = 0;
    for(i = 0; i < lines; i++)
    {
      for(j = 0; j < columns; j++)
      {
        for(k = 0; k < columns; k++)
        {
          result[i][j] = result[i][j] + (m2[i][k] * m1[k][j]);
        }
      }
    }
  }
}

void fxp_add_matrix(
  unsigned int lines,
  unsigned int columns,
  fxp_t m1[4][4],
  fxp_t m2[4][4],
  fxp_t result[4][4])
{
  unsigned int i, j;
  for(i = 0; i < lines; i++)
    for(j = 0; j < columns; j++)
      result[i][j] = fxp_add(m1[i][j], m2[i][j]);
}

void fxp_sub_matrix(
  unsigned int lines,
  unsigned int columns,
  fxp_t m1[4][4],
  fxp_t m2[4][4],
  fxp_t result[4][4])
{
  unsigned int i, j;
  for(i = 0; i < lines; i++)
    for(j = 0; j < columns; j++)
      result[i][j] = fxp_sub(m1[i][j], m2[i][j]);
}

void print_matrix(double matrix[4][4], unsigned int lines, unsigned int columns)
{
  printf("\nMatrix\n=====================\n\n");
  unsigned int i, j;
  for(i = 0; i < lines; i++)
  {
    for(j = 0; j < columns; j++)
    {
      printf("#matrix[%d][%d]: %2.2f ", i, j, matrix[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

double determinant(double a[4][4], int n)
{
  int i, j, j1, j2;
  double det = 0;
  double m[4][4];

  if(n < 1)
  {
  }
  else if(n == 1)
  {
    det = a[0][0];
  }
  else if(n == 2)
  {
    det = a[0][0] * a[1][1] - a[1][0] * a[0][1];
  }
  else
  {
    det = 0;
    for(j1 = 0; j1 < n; j1++)
    {
      for(i = 0; i < n - 1; i++)
        for(i = 1; i < n; i++)
        {
          j2 = 0;
          for(j = 0; j < n; j++)
          {
            if(j == j1)
              continue;
            m[i - 1][j2] = a[i][j];
            j2++;
          }
        }
      det +=
        internal_pow(-1.0, 1.0 + j1 + 1.0) * a[0][j1] * determinant(m, n - 1);
    }
  }
  return (det);
}

double fxp_determinant(fxp_t a_fxp[4][4], int n)
{
  int i, j, j1, j2;
  double a[4][4];

  for(i = 0; i < n; i++)
  {
    for(j = 0; j < n; j++)
    {
      a[i][j] = fxp_to_double(a_fxp[i][j]);
    }
  }

  double det = 0;
  double m[4][4];

  if(n < 1)
  {
  }
  else if(n == 1)
  {
    det = a[0][0];
  }
  else if(n == 2)
  {
    det = a[0][0] * a[1][1] - a[1][0] * a[0][1];
  }
  else
  {
    det = 0;
    for(j1 = 0; j1 < n; j1++)
    {
      for(i = 0; i < n - 1; i++)
        for(i = 1; i < n; i++)
        {
          j2 = 0;
          for(j = 0; j < n; j++)
          {
            if(j == j1)
              continue;
            m[i - 1][j2] = a[i][j];
            j2++;
          }
        }
      det +=
        internal_pow(-1.0, 1.0 + j1 + 1.0) * a[0][j1] * determinant(m, n - 1);
    }
  }
  return (det);
}

void transpose(double a[4][4], double b[4][4], int n, int m)
{
  int i, j;

  for(i = 0; i < n; i++)
  {
    for(j = 0; j < m; j++)
    {
      b[j][i] = a[i][j];
    }
  }
}

void fxp_transpose(fxp_t a[4][4], fxp_t b[4][4], int n, int m)
{
  int i, j;

  for(i = 0; i < n; i++)
  {
    for(j = 0; j < m; j++)
    {
      b[j][i] = a[i][j];
    }
  }
}
# 22 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/functions.h" 1
# 17 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/functions.h"
extern int generic_timer;
extern hardware hw;

double generic_timing_shift_l_double(double zIn, double z[], int N)
{
  generic_timer +=
    ((2 * hw.assembly.push) + (3 * hw.assembly.in) + (3 * hw.assembly.out) +
     (1 * hw.assembly.sbiw) + (1 * hw.assembly.cli) + (8 * hw.assembly.std));
  int i;
  double zOut;
  zOut = z[0];
  generic_timer +=
    ((5 * hw.assembly.ldd) + (2 * hw.assembly.mov) + (4 * hw.assembly.std) +
     (1 * hw.assembly.ld));
  generic_timer += ((2 * hw.assembly.std) + (1 * hw.assembly.rjmp));
  for(i = 0; i < N - 1; i++)
  {
    generic_timer +=
      ((17 * hw.assembly.ldd) + (4 * hw.assembly.lsl) + (4 * hw.assembly.rol) +
       (2 * hw.assembly.add) + (2 * hw.assembly.adc) + (6 * hw.assembly.mov) +
       (2 * hw.assembly.adiw) + (5 * hw.assembly.std) + (1 * hw.assembly.ld) +
       (1 * hw.assembly.st) + (1 * hw.assembly.subi) + (1 * hw.assembly.sbc) +
       (1 * hw.assembly.cp) + (1 * hw.assembly.cpc) + (1 * hw.assembly.brlt));
    z[i] = z[i + 1];
  }
  z[N - 1] = zIn;
  generic_timer +=
    ((12 * hw.assembly.ldd) + (6 * hw.assembly.mov) + (3 * hw.assembly.std) +
     (2 * hw.assembly.lsl) + (2 * hw.assembly.rol) + (1 * hw.assembly.adc) +
     (1 * hw.assembly.add) + (1 * hw.assembly.subi) + (1 * hw.assembly.sbci) +
     (1 * hw.assembly.st) + (1 * hw.assembly.adiw) + (1 * hw.assembly.in) +
     (1 * hw.assembly.cli));
  generic_timer +=
    ((3 * hw.assembly.out) + (2 * hw.assembly.pop) + (1 * hw.assembly.ret));
  return (zOut);
}

double generic_timing_shift_r_double(double zIn, double z[], int N)
{
  generic_timer +=
    ((2 * hw.assembly.push) + (3 * hw.assembly.in) + (3 * hw.assembly.out) +
     (1 * hw.assembly.sbiw) + (1 * hw.assembly.cli) + (8 * hw.assembly.std));
  int i;
  double zOut;
  zOut = z[N - 1];
  generic_timer +=
    ((7 * hw.assembly.ldd) + (2 * hw.assembly.rol) + (2 * hw.assembly.lsl) +
     (2 * hw.assembly.mov) + (4 * hw.assembly.std) + (1 * hw.assembly.add) +
     (1 * hw.assembly.adc) + (1 * hw.assembly.ld) + (1 * hw.assembly.subi) +
     (1 * hw.assembly.sbci));
  generic_timer +=
    ((2 * hw.assembly.ldd) + (2 * hw.assembly.std) + (1 * hw.assembly.sbiw) +
     (1 * hw.assembly.rjmp));
  for(i = N - 1; i > 0; i--)
  {
    z[i] = z[i - 1];
    generic_timer +=
      ((15 * hw.assembly.ldd) + (4 * hw.assembly.lsl) + (4 * hw.assembly.rol) +
       (2 * hw.assembly.add) + (2 * hw.assembly.adc) + (4 * hw.assembly.mov) +
       (5 * hw.assembly.std) + (1 * hw.assembly.subi) + (1 * hw.assembly.sbci) +
       (1 * hw.assembly.ld) + (1 * hw.assembly.st) + (1 * hw.assembly.sbiw) +
       (1 * hw.assembly.cp) + (1 * hw.assembly.cpc) + (1 * hw.assembly.brlt));
  }
  z[0] = zIn;
  generic_timer +=
    ((10 * hw.assembly.ldd) + (5 * hw.assembly.mov) + (3 * hw.assembly.std) +
     (3 * hw.assembly.out) + (2 * hw.assembly.pop) + (1 * hw.assembly.ret) +
     (1 * hw.assembly.ret) + (1 * hw.assembly.cli) + (1 * hw.assembly.in) +
     (1 * hw.assembly.st) + (1 * hw.assembly.adiw));
  return zOut;
}

fxp_t shiftL(fxp_t zIn, fxp_t z[], int N)
{
  int i;
  fxp_t zOut;
  zOut = z[0];
  for(i = 0; i < N - 1; i++)
  {
    z[i] = z[i + 1];
  }
  z[N - 1] = zIn;
  return (zOut);
}

fxp_t shiftR(fxp_t zIn, fxp_t z[], int N)
{
  int i;
  fxp_t zOut;
  zOut = z[N - 1];
  for(i = N - 1; i > 0; i--)
  {
    z[i] = z[i - 1];
  }
  z[0] = zIn;
  return zOut;
}

float shiftLfloat(float zIn, float z[], int N)
{
  int i;
  float zOut;
  zOut = z[0];
  for(i = 0; i < N - 1; i++)
  {
    z[i] = z[i + 1];
  }
  z[N - 1] = zIn;
  return (zOut);
}

float shiftRfloat(float zIn, float z[], int N)
{
  int i;
  float zOut;
  zOut = z[N - 1];
  for(i = N - 1; i > 0; i--)
  {
    z[i] = z[i - 1];
  }
  z[0] = zIn;
  return zOut;
}

double shiftRDdouble(double zIn, double z[], int N)
{
  int i;
  double zOut;
  zOut = z[0];
  for(i = 0; i < N - 1; i++)
  {
    z[i] = z[i + 1];
  }
  z[N - 1] = zIn;
  return (zOut);
}

double shiftRdouble(double zIn, double z[], int N)
{
  int i;
  double zOut;
  zOut = z[N - 1];
  for(i = N - 1; i > 0; i--)
  {
    z[i] = z[i - 1];
  }
  z[0] = zIn;
  return zOut;
}

double shiftLDouble(double zIn, double z[], int N)
{
  int i;
  double zOut;
  zOut = z[0];
  for(i = 0; i < N - 1; i++)
  {
    z[i] = z[i + 1];
  }
  z[N - 1] = zIn;
  return (zOut);
}

void shiftLboth(float zfIn, float zf[], fxp_t zIn, fxp_t z[], int N)
{
  int i;
  fxp_t zOut;
  float zfOut;
  zOut = z[0];
  zfOut = zf[0];
  for(i = 0; i < N - 1; i++)
  {
    z[i] = z[i + 1];
    zf[i] = zf[i + 1];
  }
  z[N - 1] = zIn;
  zf[N - 1] = zfIn;
}

void shiftRboth(float zfIn, float zf[], fxp_t zIn, fxp_t z[], int N)
{
  int i;
  fxp_t zOut;
  float zfOut;
  zOut = z[N - 1];
  zfOut = zf[N - 1];
  for(i = N - 1; i > 0; i--)
  {
    z[i] = z[i - 1];
    zf[i] = zf[i - 1];
  }
  z[0] = zIn;
  zf[0] = zfIn;
}

int order(int Na, int Nb)
{
  return Na > Nb ? Na - 1 : Nb - 1;
}

void fxp_check_limit_cycle(fxp_t y[], int y_size)
{
  fxp_t reference = y[y_size - 1];
  int idx = 0;
  int window_size = 1;

  for(idx = (y_size - 2); idx >= 0; idx--)
  {
    if(y[idx] != reference)
    {
      window_size++;
    }
    else
    {
      break;
    }
  }

  __DSVERIFIER_assume(window_size != y_size && window_size != 1);
  printf("window_size %d\n", window_size);
  int desired_elements = 2 * window_size;
  int found_elements = 0;

  for(idx = (y_size - 1); idx >= 0; idx--)
  {
    if(idx > (y_size - window_size - 1))
    {
      printf("%.0f == %.0f\n", y[idx], y[idx - window_size]);
      int cmp_idx = idx - window_size;
      if((cmp_idx > 0) && (y[idx] == y[idx - window_size]))
      {
        found_elements = found_elements + 2;
      }
      else
      {
        break;
      }
    }
  }
  __DSVERIFIER_assume(found_elements > 0);
  printf("desired_elements %d\n", desired_elements);
  printf("found_elements %d\n", found_elements);
  __DSVERIFIER_assume(found_elements == desired_elements);
  __DSVERIFIER_assert(0);
}

void fxp_check_persistent_limit_cycle(fxp_t *y, int y_size)
{
  int idy = 0;
  int count_same = 0;
  int window_size = 0;
  fxp_t reference = y[0];

  for(idy = 0; idy < y_size; idy++)
  {
    if(y[idy] != reference)
    {
      window_size++;
    }
    else if(window_size != 0)
    {
      break;
    }
    else
    {
      count_same++;
    }
  }
  window_size += count_same;

  __DSVERIFIER_assume(window_size > 1 && window_size <= y_size / 2);

  fxp_t lco_elements[window_size];
  for(idy = 0; idy < y_size; idy++)
  {
    if(idy < window_size)
    {
      lco_elements[idy] = y[idy];
    }
  }

  idy = 0;
  int lco_idy = 0;
  _Bool is_persistent = 0;
  while(idy < y_size)
  {
    if(y[idy++] == lco_elements[lco_idy++])
    {
      is_persistent = 1;
    }
    else
    {
      is_persistent = 0;
      break;
    }

    if(lco_idy == window_size)
    {
      lco_idy = 0;
    }
  }
  __DSVERIFIER_assert(is_persistent == 0);
}

void fxp_check_oscillations(fxp_t y[], int y_size)
{
  __DSVERIFIER_assume(
    (y[0] != y[y_size - 1]) && (y[y_size - 1] != y[y_size - 2]));
  int window_timer = 0;
  int window_count = 0;
  int i, j;
  for(i = 2; i < y_size; i++)
  {
    int window_size = i;
    for(j = 0; j < y_size; j++)
    {
      if(window_timer > window_size)
      {
        window_timer = 0;
        window_count = 0;
      }

      int window_index = j + window_size;
      if(window_index < y_size)
      {
        if(y[j] == y[window_index])
        {
          window_count++;

          __DSVERIFIER_assert(!(window_count == window_size));
        }
      }
      else
      {
        break;
      }
      window_timer++;
    }
  }
}

int fxp_ln(int x)
{
  int t, y;

  y = 0xa65af;
  if(x < 0x00008000)
    x <<= 16, y -= 0xb1721;
  if(x < 0x00800000)
    x <<= 8, y -= 0x58b91;
  if(x < 0x08000000)
    x <<= 4, y -= 0x2c5c8;
  if(x < 0x20000000)
    x <<= 2, y -= 0x162e4;
  if(x < 0x40000000)
    x <<= 1, y -= 0x0b172;
  t = x + (x >> 1);
  if((t & 0x80000000) == 0)
    x = t, y -= 0x067cd;
  t = x + (x >> 2);
  if((t & 0x80000000) == 0)
    x = t, y -= 0x03920;
  t = x + (x >> 3);
  if((t & 0x80000000) == 0)
    x = t, y -= 0x01e27;
  t = x + (x >> 4);
  if((t & 0x80000000) == 0)
    x = t, y -= 0x00f85;
  t = x + (x >> 5);
  if((t & 0x80000000) == 0)
    x = t, y -= 0x007e1;
  t = x + (x >> 6);
  if((t & 0x80000000) == 0)
    x = t, y -= 0x003f8;
  t = x + (x >> 7);
  if((t & 0x80000000) == 0)
    x = t, y -= 0x001fe;
  x = 0x80000000 - x;
  y -= x >> 15;
  return y;
}

double fxp_log10_low(double x)
{
  int xint = (int)(x * 65536.0 + 0.5);
  int lnum = fxp_ln(xint);
  int lden = fxp_ln(655360);
  return ((double)lnum / (double)lden);
}

double fxp_log10(double x)
{
  if(x > 32767.0)
  {
    if(x > 1073676289.0)
    {
      x = x / 1073676289.0;
      return fxp_log10_low(x) + 9.030873362;
    }
    x = x / 32767.0;
    return fxp_log10_low(x) + 4.515436681;
  }
  return fxp_log10_low(x);
}

float snrVariance(float s[], float n[], int blksz)
{
  int i;
  double sm = 0, nm = 0, sv = 0, nv = 0, snr;
  for(i = 0; i < blksz; i++)
  {
    sm += s[i];
    nm += n[i];
  }
  sm /= blksz;
  nm /= blksz;
  for(i = 0; i < blksz; i++)
  {
    sv += (s[i] - sm) * (s[i] - sm);
    nv += (n[i] - nm) * (n[i] - nm);
  }
  if(nv != 0.0f)
  {
    ((sv >= nv)
       ? (void)0
       : _assert(
           "sv >= nv",
           "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/functions.h",
           371));
    snr = sv / nv;
    return snr;
  }
  else
  {
    return 9999.9f;
  }
}

float snrPower(float s[], float n[], int blksz)
{
  int i;
  double sv = 0, nv = 0, snr;
  for(i = 0; i < blksz; i++)
  {
    sv += s[i] * s[i];
    nv += n[i] * n[i];
  }

  if(nv != 0.0f)
  {
    ((sv >= nv)
       ? (void)0
       : _assert(
           "sv >= nv",
           "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/functions.h",
           392));
    snr = sv / nv;
    return snr;
  }
  else
  {
    return 9999.9f;
  }
}

float snrPoint(float s[], float n[], int blksz)
{
  int i;
  double ratio = 0, power = 0;
  for(i = 0; i < blksz; i++)
  {
    if(n[i] == 0)
      continue;
    ratio = s[i] / n[i];
    if(ratio > 150.0f || ratio < -150.0f)
      continue;
    power = ratio * ratio;
    ((power >= 1.0f)
       ? (void)0
       : _assert(
           "power >= 1.0f",
           "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/functions.h",
           410));
  }

  return 9999.9f;
}

unsigned long next = 1;
int rand(void)
{
  next = next * 1103515245 + 12345;
  return (unsigned int)(next / 65536) % 32768;
}

void srand(unsigned int seed)
{
  next = seed;
}

float iirIIOutTime(float w[], float x, float a[], float b[], int Na, int Nb)
{
  int timer1 = 0;
  float *a_ptr, *b_ptr, *w_ptr;
  float sum = 0;
  a_ptr = &a[1];
  b_ptr = &b[0];
  w_ptr = &w[1];
  int k, j;
  timer1 += 71;
  for(j = 1; j < Na; j++)
  {
    w[0] -= *a_ptr++ * *w_ptr++;
    timer1 += 54;
  }
  w[0] += x;
  w_ptr = &w[0];
  for(k = 0; k < Nb; k++)
  {
    sum += *b_ptr++ * *w_ptr++;
    timer1 += 46;
  }
  timer1 += 38;
  (((double)timer1 * 1 / 16000000 <= (double)1 / 100)
     ? (void)0
     : _assert(
         "(double)timer1*CYCLE <= (double)DEADLINE",
         "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/functions.h",
         448));
  return sum;
}

float iirIItOutTime(float w[], float x, float a[], float b[], int Na, int Nb)
{
  int timer1 = 0;
  float *a_ptr, *b_ptr;
  float yout = 0;
  a_ptr = &a[1];
  b_ptr = &b[0];
  int Nw = Na > Nb ? Na : Nb;
  yout = (*b_ptr++ * x) + w[0];
  int j;
  timer1 += 105;
  for(j = 0; j < Nw - 1; j++)
  {
    w[j] = w[j + 1];
    if(j < Na - 1)
    {
      w[j] -= *a_ptr++ * yout;
      timer1 += 41;
    }
    if(j < Nb - 1)
    {
      w[j] += *b_ptr++ * x;
      timer1 += 38;
    }
    timer1 += 54;
  }
  timer1 += 7;
  (((double)timer1 * 1 / 16000000 <= (double)1 / 100)
     ? (void)0
     : _assert(
         "(double)timer1*CYCLE <= (double)DEADLINE",
         "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/functions.h",
         475));
  return yout;
}

double iirIItOutTime_double(
  double w[],
  double x,
  double a[],
  double b[],
  int Na,
  int Nb)
{
  int timer1 = 0;
  double *a_ptr, *b_ptr;
  double yout = 0;
  a_ptr = &a[1];
  b_ptr = &b[0];
  int Nw = Na > Nb ? Na : Nb;
  yout = (*b_ptr++ * x) + w[0];
  int j;
  timer1 += 105;
  for(j = 0; j < Nw - 1; j++)
  {
    w[j] = w[j + 1];
    if(j < Na - 1)
    {
      w[j] -= *a_ptr++ * yout;
      timer1 += 41;
    }
    if(j < Nb - 1)
    {
      w[j] += *b_ptr++ * x;
      timer1 += 38;
    }
    timer1 += 54;
  }
  timer1 += 7;
  (((double)timer1 * 1 / 16000000 <= (double)1 / 100)
     ? (void)0
     : _assert(
         "(double)timer1*CYCLE <= (double)DEADLINE",
         "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/functions.h",
         502));
  return yout;
}

void iirOutBoth(
  float yf[],
  float xf[],
  float af[],
  float bf[],
  float *sumf_ref,
  fxp_t y[],
  fxp_t x[],
  fxp_t a[],
  fxp_t b[],
  fxp_t *sum_ref,
  int Na,
  int Nb)
{
  fxp_t *a_ptr, *y_ptr, *b_ptr, *x_ptr;
  float *af_ptr, *yf_ptr, *bf_ptr, *xf_ptr;
  fxp_t sum = 0;
  float sumf = 0;
  a_ptr = &a[1];
  y_ptr = &y[Na - 1];
  b_ptr = &b[0];
  x_ptr = &x[Nb - 1];
  af_ptr = &af[1];
  yf_ptr = &yf[Na - 1];
  bf_ptr = &bf[0];
  xf_ptr = &xf[Nb - 1];
  int i, j;

  for(i = 0; i < Nb; i++)
  {
    sum = fxp_add(sum, fxp_mult(*b_ptr++, *x_ptr--));
    sumf += *bf_ptr++ * *xf_ptr--;
  }

  for(j = 1; j < Na; j++)
  {
    sum = fxp_sub(sum, fxp_mult(*a_ptr++, *y_ptr--));
    sumf -= *af_ptr++ * *yf_ptr--;
  }
  *sum_ref = sum;
  *sumf_ref = sumf;
}

fxp_t iirOutFixedL(
  fxp_t y[],
  fxp_t x[],
  fxp_t xin,
  fxp_t a[],
  fxp_t b[],
  int Na,
  int Nb)
{
  fxp_t *a_ptr, *y_ptr, *b_ptr, *x_ptr;
  fxp_t sum = 0;
  a_ptr = &a[Na - 1];
  y_ptr = &y[1];
  b_ptr = &b[Nb - 1];
  x_ptr = &x[0];
  int i, j;

  for(i = 0; i < Nb - 1; i++)
  {
    x[i] = x[i + 1];
    sum = fxp_add(sum, fxp_mult(*b_ptr--, *x_ptr++));
  }
  x[Nb - 1] = xin;
  sum = fxp_add(sum, fxp_mult(*b_ptr--, *x_ptr++));

  for(j = 1; j < Na - 1; j++)
  {
    sum = fxp_sub(sum, fxp_mult(*a_ptr--, *y_ptr++));
    y[j] = y[j + 1];
  }
  if(Na > 1)
    sum = fxp_sub(sum, fxp_mult(*a_ptr--, *y_ptr++));
  y[Na - 1] = sum;
  return sum;
}

float iirOutFloatL(
  float y[],
  float x[],
  float xin,
  float a[],
  float b[],
  int Na,
  int Nb)
{
  float *a_ptr, *y_ptr, *b_ptr, *x_ptr;
  float sum = 0;
  a_ptr = &a[Na - 1];
  y_ptr = &y[1];
  b_ptr = &b[Nb - 1];
  x_ptr = &x[0];
  int i, j;

  for(i = 0; i < Nb - 1; i++)
  {
    x[i] = x[i + 1];
    sum += *b_ptr-- * *x_ptr++;
  }
  x[Nb - 1] = xin;
  sum += *b_ptr-- * *x_ptr++;

  for(j = 1; j < Na - 1; j++)
  {
    sum -= *a_ptr-- * *y_ptr++;
    y[j] = y[j + 1];
  }
  if(Na > 1)
    sum -= *a_ptr-- * *y_ptr++;
  y[Na - 1] = sum;
  return sum;
}

float iirOutBothL(
  float yf[],
  float xf[],
  float af[],
  float bf[],
  float xfin,
  fxp_t y[],
  fxp_t x[],
  fxp_t a[],
  fxp_t b[],
  fxp_t xin,
  int Na,
  int Nb)
{
  fxp_t *a_ptr, *y_ptr, *b_ptr, *x_ptr;
  fxp_t sum = 0;
  a_ptr = &a[Na - 1];
  y_ptr = &y[1];
  b_ptr = &b[Nb - 1];
  x_ptr = &x[0];
  float *af_ptr, *yf_ptr, *bf_ptr, *xf_ptr;
  float sumf = 0;
  af_ptr = &af[Na - 1];
  yf_ptr = &yf[1];
  bf_ptr = &bf[Nb - 1];
  xf_ptr = &xf[0];
  int i, j;

  for(i = 0; i < Nb - 1; i++)
  {
    x[i] = x[i + 1];
    sum = fxp_add(sum, fxp_mult(*b_ptr--, *x_ptr++));
    xf[i] = xf[i + 1];
    sumf += *bf_ptr-- * *xf_ptr++;
  }
  x[Nb - 1] = xin;
  sum = fxp_add(sum, fxp_mult(*b_ptr--, *x_ptr++));
  xf[Nb - 1] = xfin;
  sumf += *bf_ptr-- * *xf_ptr++;

  for(j = 1; j < Na - 1; j++)
  {
    sum = fxp_sub(sum, fxp_mult(*a_ptr--, *y_ptr++));
    y[j] = y[j + 1];
    sumf -= *af_ptr-- * *yf_ptr++;
    yf[j] = yf[j + 1];
  }
  if(Na > 1)
    sum = fxp_sub(sum, fxp_mult(*a_ptr--, *y_ptr++));
  y[Na - 1] = sum;
  if(Na > 1)
    sumf -= *af_ptr-- * *yf_ptr++;
  yf[Na - 1] = sumf;
  return fxp_to_float(sum) - sumf;
}

float iirOutBothL2(
  float yf[],
  float xf[],
  float af[],
  float bf[],
  float xfin,
  fxp_t y[],
  fxp_t x[],
  fxp_t a[],
  fxp_t b[],
  fxp_t xin,
  int Na,
  int Nb)
{
  fxp_t *a_ptr, *y_ptr, *b_ptr, *x_ptr;
  fxp_t sum = 0;
  a_ptr = &a[Na - 1];
  y_ptr = &y[1];
  b_ptr = &b[Nb - 1];
  x_ptr = &x[0];
  float *af_ptr, *yf_ptr, *bf_ptr, *xf_ptr;
  float sumf = 0;
  af_ptr = &af[Na - 1];
  yf_ptr = &yf[1];
  bf_ptr = &bf[Nb - 1];
  xf_ptr = &xf[0];
  int i = 0, j = 1;

  for(i = 0; i < Nb - 1; i++)
  {
    x[i] = x[i + 1];
    sum = fxp_add(sum, fxp_mult(b[Nb - 1 - i], x[i]));
    xf[i] = xf[i + 1];
    sumf += bf[Nb - 1 - i] * xf[i];
  }
  x[Nb - 1] = xin;
  sum = fxp_add(sum, fxp_mult(b[Nb - 1 - i], x[i]));
  xf[Nb - 1] = xfin;
  sumf += bf[Nb - 1 - i] * xf[i];

  for(j = 1; j < Na - 1; j++)
  {
    sum = fxp_sub(sum, fxp_mult(a[Na - j], y[j]));
    y[j] = y[j + 1];
    sumf -= af[Na - j] * yf[j];
    yf[j] = yf[j + 1];
  }
  if(Na > 1)
    sum = fxp_sub(sum, fxp_mult(a[Na - j], y[j]));
  y[Na - 1] = sum;
  if(Na > 1)
    sumf -= af[Na - j] * yf[j];
  yf[Na - 1] = sumf;
  return fxp_to_float(sum) - sumf;
}
# 23 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/realizations.h" 1
# 17 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/realizations.h"
extern digital_system ds;
extern hardware hw;
extern int generic_timer;

fxp_t fxp_direct_form_1(
  fxp_t y[],
  fxp_t x[],
  fxp_t a[],
  fxp_t b[],
  int Na,
  int Nb)
{
  fxp_t *a_ptr, *y_ptr, *b_ptr, *x_ptr;
  fxp_t sum = 0;
  a_ptr = &a[1];
  y_ptr = &y[Na - 1];
  b_ptr = &b[0];
  x_ptr = &x[Nb - 1];
  int i, j;
  for(i = 0; i < Nb; i++)
  {
    sum = fxp_add(sum, fxp_mult(*b_ptr++, *x_ptr--));
  }

  for(j = 1; j < Na; j++)
  {
    sum = fxp_sub(sum, fxp_mult(*a_ptr++, *y_ptr--));
  }
  sum = fxp_div(sum, a[0]);
  return fxp_quantize(sum);
}

fxp_t fxp_direct_form_2(
  fxp_t w[],
  fxp_t x,
  fxp_t a[],
  fxp_t b[],
  int Na,
  int Nb)
{
  fxp_t *a_ptr, *b_ptr, *w_ptr;
  fxp_t sum = 0;
  a_ptr = &a[1];
  b_ptr = &b[0];
  w_ptr = &w[1];
  int k, j;
  for(j = 1; j < Na; j++)
  {
    w[0] = fxp_sub(w[0], fxp_mult(*a_ptr++, *w_ptr++));
  }
  w[0] = fxp_add(w[0], x);
  w[0] = fxp_div(w[0], a[0]);

  w_ptr = &w[0];
  for(k = 0; k < Nb; k++)
  {
    sum = fxp_add(sum, fxp_mult(*b_ptr++, *w_ptr++));
  }

  return fxp_quantize(sum);
}

fxp_t fxp_transposed_direct_form_2(
  fxp_t w[],
  fxp_t x,
  fxp_t a[],
  fxp_t b[],
  int Na,
  int Nb)
{
  fxp_t *a_ptr, *b_ptr;
  fxp_t yout = 0;
  a_ptr = &a[1];
  b_ptr = &b[0];
  int Nw = Na > Nb ? Na : Nb;
  yout = fxp_add(fxp_mult(*b_ptr++, x), w[0]);
  yout = fxp_div(yout, a[0]);
  int j;
  for(j = 0; j < Nw - 1; j++)
  {
    w[j] = w[j + 1];
    if(j < Na - 1)
    {
      w[j] = fxp_sub(w[j], fxp_mult(*a_ptr++, yout));
    }
    if(j < Nb - 1)
    {
      w[j] = fxp_add(w[j], fxp_mult(*b_ptr++, x));
    }
  }

  return fxp_quantize(yout);
}

double double_direct_form_1(
  double y[],
  double x[],
  double a[],
  double b[],
  int Na,
  int Nb)
{
  double *a_ptr, *y_ptr, *b_ptr, *x_ptr;
  double sum = 0;
  a_ptr = &a[1];
  y_ptr = &y[Na - 1];
  b_ptr = &b[0];
  x_ptr = &x[Nb - 1];
  int i, j;
  for(i = 0; i < Nb; i++)
  {
    sum += *b_ptr++ * *x_ptr--;
  }
  for(j = 1; j < Na; j++)
  {
    sum -= *a_ptr++ * *y_ptr--;
  }
  sum = (sum / a[0]);
  return sum;
}

double double_direct_form_2(
  double w[],
  double x,
  double a[],
  double b[],
  int Na,
  int Nb)
{
  double *a_ptr, *b_ptr, *w_ptr;
  double sum = 0;
  a_ptr = &a[1];
  b_ptr = &b[0];
  w_ptr = &w[1];
  int k, j;
  for(j = 1; j < Na; j++)
  {
    w[0] -= *a_ptr++ * *w_ptr++;
  }
  w[0] += x;
  w[0] = w[0] / a[0];
  w_ptr = &w[0];
  for(k = 0; k < Nb; k++)
  {
    sum += *b_ptr++ * *w_ptr++;
  }
  return sum;
}

double double_transposed_direct_form_2(
  double w[],
  double x,
  double a[],
  double b[],
  int Na,
  int Nb)
{
  double *a_ptr, *b_ptr;
  double yout = 0;
  a_ptr = &a[1];
  b_ptr = &b[0];
  int Nw = Na > Nb ? Na : Nb;
  yout = (*b_ptr++ * x) + w[0];
  yout = yout / a[0];
  int j;
  for(j = 0; j < Nw - 1; j++)
  {
    w[j] = w[j + 1];
    if(j < Na - 1)
    {
      w[j] -= *a_ptr++ * yout;
    }
    if(j < Nb - 1)
    {
      w[j] += *b_ptr++ * x;
    }
  }
  return yout;
}

float float_direct_form_1(
  float y[],
  float x[],
  float a[],
  float b[],
  int Na,
  int Nb)
{
  float *a_ptr, *y_ptr, *b_ptr, *x_ptr;
  float sum = 0;
  a_ptr = &a[1];
  y_ptr = &y[Na - 1];
  b_ptr = &b[0];
  x_ptr = &x[Nb - 1];
  int i, j;
  for(i = 0; i < Nb; i++)
  {
    sum += *b_ptr++ * *x_ptr--;
  }
  for(j = 1; j < Na; j++)
  {
    sum -= *a_ptr++ * *y_ptr--;
  }
  sum = (sum / a[0]);
  return sum;
}

float float_direct_form_2(
  float w[],
  float x,
  float a[],
  float b[],
  int Na,
  int Nb)
{
  float *a_ptr, *b_ptr, *w_ptr;
  float sum = 0;
  a_ptr = &a[1];
  b_ptr = &b[0];
  w_ptr = &w[1];
  int k, j;
  for(j = 1; j < Na; j++)
  {
    w[0] -= *a_ptr++ * *w_ptr++;
  }
  w[0] += x;
  w[0] = w[0] / a[0];
  w_ptr = &w[0];
  for(k = 0; k < Nb; k++)
  {
    sum += *b_ptr++ * *w_ptr++;
  }
  return sum;
}

float float_transposed_direct_form_2(
  float w[],
  float x,
  float a[],
  float b[],
  int Na,
  int Nb)
{
  float *a_ptr, *b_ptr;
  float yout = 0;
  a_ptr = &a[1];
  b_ptr = &b[0];
  int Nw = Na > Nb ? Na : Nb;
  yout = (*b_ptr++ * x) + w[0];
  yout = yout / a[0];
  int j;
  for(j = 0; j < Nw - 1; j++)
  {
    w[j] = w[j + 1];
    if(j < Na - 1)
    {
      w[j] -= *a_ptr++ * yout;
    }
    if(j < Nb - 1)
    {
      w[j] += *b_ptr++ * x;
    }
  }
  return yout;
}

double double_direct_form_1_MSP430(
  double y[],
  double x[],
  double a[],
  double b[],
  int Na,
  int Nb)
{
  int timer1 = 0;
  double *a_ptr, *y_ptr, *b_ptr, *x_ptr;
  double sum = 0;
  a_ptr = &a[1];
  y_ptr = &y[Na - 1];
  b_ptr = &b[0];
  x_ptr = &x[Nb - 1];
  int i, j;
  timer1 += 91;
  for(i = 0; i < Nb; i++)
  {
    sum += *b_ptr++ * *x_ptr--;
    timer1 += 47;
  }
  for(j = 1; j < Na; j++)
  {
    sum -= *a_ptr++ * *y_ptr--;
    timer1 += 57;
  }
  timer1 += 3;
  (((double)timer1 * hw.cycle <= ds.sample_time)
     ? (void)0
     : _assert(
         "(double) timer1 * hw.cycle <= ds.sample_time",
         "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/realizations.h",
         229));
  return sum;
}

double double_direct_form_2_MSP430(
  double w[],
  double x,
  double a[],
  double b[],
  int Na,
  int Nb)
{
  int timer1 = 0;
  double *a_ptr, *b_ptr, *w_ptr;
  double sum = 0;
  a_ptr = &a[1];
  b_ptr = &b[0];
  w_ptr = &w[1];
  int k, j;
  timer1 += 71;
  for(j = 1; j < Na; j++)
  {
    w[0] -= *a_ptr++ * *w_ptr++;
    timer1 += 54;
  }
  w[0] += x;
  w[0] = w[0] / a[0];
  w_ptr = &w[0];
  for(k = 0; k < Nb; k++)
  {
    sum += *b_ptr++ * *w_ptr++;
    timer1 += 46;
  }
  timer1 += 38;
  (((double)timer1 * hw.cycle <= ds.sample_time)
     ? (void)0
     : _assert(
         "(double) timer1 * hw.cycle <= ds.sample_time",
         "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/realizations.h",
         256));
  return sum;
}

double double_transposed_direct_form_2_MSP430(
  double w[],
  double x,
  double a[],
  double b[],
  int Na,
  int Nb)
{
  int timer1 = 0;
  double *a_ptr, *b_ptr;
  double yout = 0;
  a_ptr = &a[1];
  b_ptr = &b[0];
  int Nw = Na > Nb ? Na : Nb;
  yout = (*b_ptr++ * x) + w[0];
  int j;
  timer1 += 105;
  for(j = 0; j < Nw - 1; j++)
  {
    w[j] = w[j + 1];
    if(j < Na - 1)
    {
      w[j] -= *a_ptr++ * yout;
      timer1 += 41;
    }
    if(j < Nb - 1)
    {
      w[j] += *b_ptr++ * x;
      timer1 += 38;
    }
    timer1 += 54;
  }
  timer1 += 7;
  (((double)timer1 * hw.cycle <= ds.sample_time)
     ? (void)0
     : _assert(
         "(double) timer1 * hw.cycle <= ds.sample_time",
         "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/realizations.h",
         285));
  return yout;
}

double generic_timing_double_direct_form_1(
  double y[],
  double x[],
  double a[],
  double b[],
  int Na,
  int Nb)
{
  generic_timer +=
    ((6 * hw.assembly.push) + (3 * hw.assembly.in) + (1 * hw.assembly.sbiw) +
     (1 * hw.assembly.cli) + (3 * hw.assembly.out) + (12 * hw.assembly.std));
  double *a_ptr, *y_ptr, *b_ptr, *x_ptr;
  double sum = 0;
  a_ptr = &a[1];
  y_ptr = &y[Na - 1];
  b_ptr = &b[0];
  x_ptr = &x[Nb - 1];
  generic_timer +=
    ((12 * hw.assembly.std) + (12 * hw.assembly.ldd) + (2 * hw.assembly.subi) +
     (2 * hw.assembly.sbci) + (4 * hw.assembly.lsl) + (4 * hw.assembly.rol) +
     (2 * hw.assembly.add) + (2 * hw.assembly.adc) + (1 * hw.assembly.adiw));
  int i, j;
  generic_timer += ((2 * hw.assembly.std) + (1 * hw.assembly.rjmp));
  for(i = 0; i < Nb; i++)
  {
    generic_timer +=
      ((20 * hw.assembly.ldd) + (24 * hw.assembly.mov) +
       (2 * hw.assembly.subi) + (1 * hw.assembly.sbci) + (1 * hw.assembly.sbc) +
       (10 * hw.assembly.std) + (2 * hw.assembly.ld) + (2 * hw.assembly.rcall) +
       (1 * hw.assembly.adiw) + (1 * hw.assembly.cp) + (1 * hw.assembly.cpc) +
       (1 * hw.assembly.adiw) + (1 * hw.assembly.brge) +
       (1 * hw.assembly.rjmp));
    sum += *b_ptr++ * *x_ptr--;
  }
  generic_timer +=
    ((2 * hw.assembly.ldi) + (2 * hw.assembly.std) + (1 * hw.assembly.rjmp));
  for(j = 1; j < Na; j++)
  {
    generic_timer +=
      ((22 * hw.assembly.ldd) + (24 * hw.assembly.mov) +
       (2 * hw.assembly.subi) + (8 * hw.assembly.std) + (1 * hw.assembly.sbci) +
       (2 * hw.assembly.ld) + (2 * hw.assembly.rcall) + (1 * hw.assembly.sbc) +
       (1 * hw.assembly.adiw) + (1 * hw.assembly.cp) + (1 * hw.assembly.cpc) +
       (1 * hw.assembly.adiw) + (1 * hw.assembly.brge) +
       (1 * hw.assembly.rjmp));
    sum -= *a_ptr++ * *y_ptr--;
  }
  generic_timer +=
    ((4 * hw.assembly.ldd) + (4 * hw.assembly.mov) + (1 * hw.assembly.adiw) +
     (1 * hw.assembly.in) + (1 * hw.assembly.cli) + (3 * hw.assembly.out) +
     (6 * hw.assembly.pop) + (1 * hw.assembly.ret));
  return sum;
}

double generic_timing_double_direct_form_2(
  double w[],
  double x,
  double a[],
  double b[],
  int Na,
  int Nb)
{
  generic_timer +=
    ((8 * hw.assembly.push) + (14 * hw.assembly.std) + (3 * hw.assembly.out) +
     (3 * hw.assembly.in) + (1 * hw.assembly.sbiw) + (1 * hw.assembly.cli));
  double *a_ptr, *b_ptr, *w_ptr;
  double sum = 0;
  a_ptr = &a[1];
  b_ptr = &b[0];
  w_ptr = &w[1];
  int k, j;
  generic_timer +=
    ((10 * hw.assembly.std) + (6 * hw.assembly.ldd) + (2 * hw.assembly.adiw));
  generic_timer +=
    ((2 * hw.assembly.ldi) + (2 * hw.assembly.std) + (1 * hw.assembly.rjmp));
  for(j = 1; j < Na; j++)
  {
    w[0] -= *a_ptr++ * *w_ptr++;
    generic_timer +=
      ((23 * hw.assembly.ldd) + (32 * hw.assembly.mov) + (9 * hw.assembly.std) +
       (2 * hw.assembly.subi) + (3 * hw.assembly.ld) + (2 * hw.assembly.rcall) +
       (2 * hw.assembly.sbci) + (1 * hw.assembly.st) + (1 * hw.assembly.adiw) +
       (1 * hw.assembly.cp) + (1 * hw.assembly.cpc) + (1 * hw.assembly.brge));
  }
  w[0] += x;
  w_ptr = &w[0];
  generic_timer +=
    ((13 * hw.assembly.ldd) + (12 * hw.assembly.mov) + (5 * hw.assembly.std) +
     (1 * hw.assembly.st) + (1 * hw.assembly.ld) + (1 * hw.assembly.rcall));
  generic_timer += ((2 * hw.assembly.std) + (1 * hw.assembly.rjmp));
  for(k = 0; k < Nb; k++)
  {
    sum += *b_ptr++ * *w_ptr++;
    generic_timer +=
      ((20 * hw.assembly.ldd) + (24 * hw.assembly.mov) +
       (10 * hw.assembly.std) + (2 * hw.assembly.rcall) + (2 * hw.assembly.ld) +
       (2 * hw.assembly.subi) + (2 * hw.assembly.sbci) +
       (1 * hw.assembly.adiw) + (1 * hw.assembly.cp) + (1 * hw.assembly.cpc) +
       (1 * hw.assembly.brge) + (1 * hw.assembly.rjmp));
  }
  generic_timer +=
    ((4 * hw.assembly.ldd) + (4 * hw.assembly.mov) + (1 * hw.assembly.adiw) +
     (1 * hw.assembly.in) + (1 * hw.assembly.cli) + (3 * hw.assembly.out) +
     (8 * hw.assembly.pop) + (1 * hw.assembly.ret));
  return sum;
}

double generic_timing_double_transposed_direct_form_2(
  double w[],
  double x,
  double a[],
  double b[],
  int Na,
  int Nb)
{
  generic_timer +=
    ((8 * hw.assembly.push) + (14 * hw.assembly.std) + (3 * hw.assembly.out) +
     (3 * hw.assembly.in) + (1 * hw.assembly.sbiw) + (1 * hw.assembly.cli));
  double *a_ptr, *b_ptr;
  double yout = 0;
  a_ptr = &a[1];
  b_ptr = &b[0];
  int Nw = Na > Nb ? Na : Nb;
  yout = (*b_ptr++ * x) + w[0];
  int j;
  generic_timer +=
    ((15 * hw.assembly.std) + (22 * hw.assembly.ldd) + (24 * hw.assembly.mov) +
     (2 * hw.assembly.rcall) + (2 * hw.assembly.ld) + (1 * hw.assembly.cp) +
     (1 * hw.assembly.cpc) + (1 * hw.assembly.subi) + (1 * hw.assembly.sbci) +
     (1 * hw.assembly.brge) + (1 * hw.assembly.adiw));
  generic_timer += ((2 * hw.assembly.std) + (1 * hw.assembly.rjmp));
  for(j = 0; j < Nw - 1; j++)
  {
    w[j] = w[j + 1];
    if(j < Na - 1)
    {
      w[j] -= *a_ptr++ * yout;
    }
    if(j < Nb - 1)
    {
      w[j] += *b_ptr++ * x;
    }
    generic_timer +=
      ((70 * hw.assembly.mov) + (65 * hw.assembly.ldd) +
       (12 * hw.assembly.lsl) + (12 * hw.assembly.rol) +
       (15 * hw.assembly.std) + (6 * hw.assembly.add) + (6 * hw.assembly.adc) +
       (2 * hw.assembly.adiw) + (3 * hw.assembly.cpc) + (3 * hw.assembly.cp) +
       (5 * hw.assembly.ld) + (4 * hw.assembly.rcall) + (5 * hw.assembly.subi) +
       (3 * hw.assembly.rjmp) + (2 * hw.assembly.brlt) + (3 * hw.assembly.st) +
       (2 * hw.assembly.sbci) + (3 * hw.assembly.sbc) + (1 * hw.assembly.brge));
  }
  generic_timer +=
    ((4 * hw.assembly.ldd) + (4 * hw.assembly.mov) + (8 * hw.assembly.pop) +
     (3 * hw.assembly.out) + (1 * hw.assembly.in) + (1 * hw.assembly.cli) +
     (1 * hw.assembly.adiw) + (1 * hw.assembly.ret));
  return yout;
}

void double_direct_form_1_impl2(
  double x[],
  int x_size,
  double b[],
  int b_size,
  double a[],
  int a_size,
  double y[])
{
  int i = 0;
  int j = 0;

  double v[x_size];
  for(i = 0; i < x_size; i++)
  {
    v[i] = 0;
    for(j = 0; j < b_size; j++)
    {
      if(j > i)
        break;
      v[i] = v[i] + x[i - j] * b[j];
    }
  }

  y[0] = v[0];
  for(i = 1; i < x_size; i++)
  {
    y[i] = 0;
    y[i] = y[i] + v[i];
    for(j = 1; j < a_size; j++)
    {
      if(j > i)
        break;
      y[i] = y[i] + y[i - j] * ((-1) * a[j]);
    }
  }
}

void fxp_direct_form_1_impl2(
  fxp_t x[],
  int x_size,
  fxp_t b[],
  int b_size,
  fxp_t a[],
  int a_size,
  fxp_t y[])
{
  int i = 0;
  int j = 0;

  fxp_t v[x_size];
  for(i = 0; i < x_size; i++)
  {
    v[i] = 0;
    for(j = 0; j < b_size; j++)
    {
      if(j > i)
        break;
      v[i] = fxp_add(v[i], fxp_mult(x[i - j], b[j]));
    }
  }

  y[0] = v[0];
  for(i = 1; i < x_size; i++)
  {
    y[i] = 0;
    y[i] = fxp_add(y[i], v[i]);
    for(j = 1; j < a_size; j++)
    {
      if(j > i)
        break;
      y[i] = fxp_add(y[i], fxp_mult(y[i - j], -a[j]));
    }
  }
}
# 24 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/delta-operator.h" 1
# 17 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/delta-operator.h"
# 1 "c:\\tools\\mingw\\mingw-0.6.2\\include\\assert.h" 1 3
# 38 "c:\\tools\\mingw\\mingw-0.6.2\\include\\assert.h" 3
void __attribute__((__cdecl__)) __attribute__((__nothrow__))
_assert(const char *, const char *, int) __attribute__((__noreturn__));
# 18 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/delta-operator.h" 2

# 1 "c:\\tools\\mingw\\mingw-0.6.2\\include\\assert.h" 1 3
# 38 "c:\\tools\\mingw\\mingw-0.6.2\\include\\assert.h" 3
void __attribute__((__cdecl__)) __attribute__((__nothrow__))
_assert(const char *, const char *, int) __attribute__((__noreturn__));
# 21 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/delta-operator.h" 2

int nchoosek(int n, int k)
{
  if(k == 0)
    return 1;

  return (n * nchoosek(n - 1, k - 1)) / k;
}

void generate_delta_coefficients(
  double vetor[],
  double out[],
  int n,
  double delta)
{
  int i, j;
  int N = n - 1;
  double sum_delta_operator;

  for(i = 0; i <= N; i++)
  {
    sum_delta_operator = 0;
    for(j = 0; j <= i; j++)
    {
      sum_delta_operator =
        sum_delta_operator + vetor[j] * nchoosek(N - j, i - j);
    }
    out[i] = internal_pow(delta, N - i) * sum_delta_operator;
  }
}

void get_delta_transfer_function(
  double b[],
  double b_out[],
  int b_size,
  double a[],
  double a_out[],
  int a_size,
  double delta)
{
  generate_delta_coefficients(b, b_out, b_size, delta);
  generate_delta_coefficients(a, a_out, a_size, delta);
}

void get_delta_transfer_function_with_base(
  double b[],
  double b_out[],
  int b_size,
  double a[],
  double a_out[],
  int a_size,
  double delta)
{
  int i, j;
  int N = a_size - 1;
  int M = b_size - 1;
  double sum_delta_operator;

  for(i = 0; i <= N; i++)
  {
    sum_delta_operator = 0;
    for(j = 0; j <= i; j++)
    {
      sum_delta_operator = sum_delta_operator + a[j] * nchoosek(N - j, i - j);
    }
    a_out[i] = internal_pow(delta, N - i) * sum_delta_operator;
  }

  for(i = 0; i <= M; i++)
  {
    sum_delta_operator = 0;
    for(j = 0; j <= i; j++)
    {
      sum_delta_operator = sum_delta_operator + b[j] * nchoosek(M - j, i - j);
    }
    b_out[i] = internal_pow(delta, M - i) * sum_delta_operator;
  }
}
# 25 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/closed-loop.h" 1
# 28 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/closed-loop.h"
void ft_closedloop_series(
  double c_num[],
  int Nc_num,
  double c_den[],
  int Nc_den,
  double model_num[],
  int Nmodel_num,
  double model_den[],
  int Nmodel_den,
  double ans_num[],
  int Nans_num,
  double ans_den[],
  int Nans_den)
{
  Nans_num = Nc_num + Nmodel_num - 1;
  Nans_den = Nc_den + Nmodel_den - 1;
  double den_mult[Nans_den];
  poly_mult(c_num, Nc_num, model_num, Nmodel_num, ans_num, Nans_num);
  poly_mult(c_den, Nc_den, model_den, Nmodel_den, den_mult, Nans_den);
  poly_sum(ans_num, Nans_num, den_mult, Nans_den, ans_den, Nans_den);
}

void ft_closedloop_sensitivity(
  double c_num[],
  int Nc_num,
  double c_den[],
  int Nc_den,
  double model_num[],
  int Nmodel_num,
  double model_den[],
  int Nmodel_den,
  double ans_num[],
  int Nans_num,
  double ans_den[],
  int Nans_den)
{
  int Nans_num_p = Nc_num + Nmodel_num - 1;
  Nans_den = Nc_den + Nmodel_den - 1;
  Nans_num = Nc_den + Nmodel_den - 1;
  double num_mult[Nans_num_p];
  poly_mult(c_den, Nc_den, model_den, Nmodel_den, ans_num, Nans_num);
  poly_mult(c_num, Nc_num, model_num, Nmodel_num, num_mult, Nans_num_p);
  poly_sum(ans_num, Nans_num, num_mult, Nans_num_p, ans_den, Nans_den);
}

void ft_closedloop_feedback(
  double c_num[],
  int Nc_num,
  double c_den[],
  int Nc_den,
  double model_num[],
  int Nmodel_num,
  double model_den[],
  int Nmodel_den,
  double ans_num[],
  int Nans_num,
  double ans_den[],
  int Nans_den)
{
  Nans_num = Nc_den + Nmodel_num - 1;
  Nans_den = Nc_den + Nmodel_den - 1;
  int Nnum_mult = Nc_num + Nmodel_num - 1;
  double den_mult[Nans_den];
  double num_mult[Nnum_mult];
  poly_mult(c_num, Nc_num, model_num, Nmodel_num, num_mult, Nnum_mult);
  poly_mult(c_den, Nc_den, model_den, Nmodel_den, den_mult, Nans_den);
  poly_sum(num_mult, Nnum_mult, den_mult, Nans_den, ans_den, Nans_den);
  poly_mult(c_den, Nc_den, model_num, Nmodel_num, ans_num, Nans_num);
}

int check_stability_closedloop(
  double a[],
  int n,
  double plant_num[],
  int p_num_size,
  double plant_den[],
  int p_den_size)
{
  int columns = n;
  double m[2 * n - 1][n];
  int i, j;
  int first_is_positive = 0;
  double *p_num = plant_num;
  double *p_den = plant_den;

  double sum = 0;
  for(i = 0; i < n; i++)
  {
    sum += a[i];
  }
  __DSVERIFIER_assert(sum > 0);

  sum = 0;
  for(i = 0; i < n; i++)
  {
    sum += a[i] * internal_pow(-1, n - 1 - i);
  }
  sum = sum * internal_pow(-1, n - 1);
  __DSVERIFIER_assert(sum > 0);

  __DSVERIFIER_assert(internal_abs(a[n - 1]) < a[0]);

  for(i = 0; i < 2 * n - 1; i++)
  {
    for(j = 0; j < columns; j++)
    {
      m[i][j] = 0;
      if(i == 0)
      {
        m[i][j] = a[j];
        continue;
      }
      if(i % 2 != 0)
      {
        int x;
        for(x = 0; x < columns; x++)
        {
          m[i][x] = m[i - 1][columns - x - 1];
        }
        columns = columns - 1;
        j = columns;
      }
      else
      {
        m[i][j] = m[i - 2][j] - (m[i - 2][columns] / m[i - 2][0]) * m[i - 1][j];
        __DSVERIFIER_assert((m[0][0] >= 0) && (m[i][0] >= 0));
      }
    }
  }
  return 1;
}
# 26 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/initialization.h" 1
# 17 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/initialization.h"
extern digital_system ds;
extern digital_system plant;
extern digital_system control;
extern implementation impl;
extern hardware hw;

void initialization()
{
  if(impl.frac_bits >= 32)
  {
    printf("impl.frac_bits must be less than word width!\n");
  }
  if(impl.int_bits >= 32 - impl.frac_bits)
  {
    printf(
      "impl.int_bits must be less than word width subtracted by precision!\n");
    ((0)
       ? (void)0
       : _assert(
           "0",
           "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/initialization.h",
           30));
  }
  if(impl.frac_bits >= 31)
  {
    _fxp_one = 0x7fffffff;
  }
  else
  {
    _fxp_one = (0x00000001 << impl.frac_bits);
  }

  _fxp_half = (0x00000001 << (impl.frac_bits - 1));
  _fxp_minus_one = -(0x00000001 << impl.frac_bits);
  _fxp_min = -(0x00000001 << (impl.frac_bits + impl.int_bits - 1));
  _fxp_max = (0x00000001 << (impl.frac_bits + impl.int_bits - 1)) - 1;
  _fxp_fmask = ((((int32_t)1) << impl.frac_bits) - 1);
  _fxp_imask = ((0x80000000) >> (32 - impl.frac_bits - 1));

  _dbl_min = _fxp_min;
  _dbl_min /= (1 << impl.frac_bits);
  _dbl_max = _fxp_max;
  _dbl_max /= (1 << impl.frac_bits);

  if((impl.scale == 0) || (impl.scale == 1))
  {
    impl.scale = 1;
    return;
  }

  if(impl.min != 0)
  {
    impl.min = impl.min / impl.scale;
  }
  if(impl.max != 0)
  {
    impl.max = impl.max / impl.scale;
  }
# 77 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/initialization.h"
}
# 27 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/state-space.h" 1
# 17 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/core/state-space.h"
extern digital_system_state_space _controller;

extern int nStates;
extern int nInputs;
extern int nOutputs;

double double_state_space_representation(void)
{
  double result1[4][4];
  double result2[4][4];

  int i, j;
  for(i = 0; i < 4; i++)
  {
    for(j = 0; j < 4; j++)
    {
      result1[i][j] = 0;
      result2[i][j] = 0;
    }
  }

  double_matrix_multiplication(
    nOutputs, nStates, nStates, 1, _controller.C, _controller.states, result1);
  double_matrix_multiplication(
    nOutputs, nInputs, nInputs, 1, _controller.D, _controller.inputs, result2);

  double_add_matrix(nOutputs, 1, result1, result2, _controller.outputs);

  for(i = 1; i < 7; i++)
  {
    double_matrix_multiplication(
      nStates, nStates, nStates, 1, _controller.A, _controller.states, result1);
    double_matrix_multiplication(
      nStates, nInputs, nInputs, 1, _controller.B, _controller.inputs, result2);

    double_add_matrix(nStates, 1, result1, result2, _controller.states);

    double_matrix_multiplication(
      nOutputs,
      nStates,
      nStates,
      1,
      _controller.C,
      _controller.states,
      result1);
    double_matrix_multiplication(
      nOutputs,
      nInputs,
      nInputs,
      1,
      _controller.D,
      _controller.inputs,
      result2);

    double_add_matrix(nOutputs, 1, result1, result2, _controller.outputs);
  }
  return _controller.outputs[0][0];
}

double fxp_state_space_representation(void)
{
  fxp_t result1[4][4];
  fxp_t result2[4][4];

  int i, j;
  for(i = 0; i < 4; i++)
  {
    for(j = 0; j < 4; j++)
    {
      result1[i][j] = 0;
      result2[i][j] = 0;
    }
  }

  fxp_t A_fpx[4][4];
  fxp_t B_fpx[4][4];
  fxp_t C_fpx[4][4];
  fxp_t D_fpx[4][4];
  fxp_t states_fpx[4][4];
  fxp_t inputs_fpx[4][4];
  fxp_t outputs_fpx[4][4];

  for(i = 0; i < 4; i++)
  {
    for(j = 0; j < 4; j++)
    {
      A_fpx[i][j] = 0;
    }
  }

  for(i = 0; i < 4; i++)
  {
    for(j = 0; j < 4; j++)
    {
      B_fpx[i][j] = 0;
    }
  }

  for(i = 0; i < 4; i++)
  {
    for(j = 0; j < 4; j++)
    {
      C_fpx[i][j] = 0;
    }
  }

  for(i = 0; i < 4; i++)
  {
    for(j = 0; j < 4; j++)
    {
      D_fpx[i][j] = 0;
    }
  }

  for(i = 0; i < 4; i++)
  {
    for(j = 0; j < 4; j++)
    {
      states_fpx[i][j] = 0;
    }
  }

  for(i = 0; i < 4; i++)
  {
    for(j = 0; j < 4; j++)
    {
      inputs_fpx[i][j] = 0;
    }
  }

  for(i = 0; i < 4; i++)
  {
    for(j = 0; j < 4; j++)
    {
      outputs_fpx[i][j] = 0;
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < nStates; j++)
    {
      A_fpx[i][j] = fxp_double_to_fxp(_controller.A[i][j]);
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < nInputs; j++)
    {
      B_fpx[i][j] = fxp_double_to_fxp(_controller.B[i][j]);
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < nStates; j++)
    {
      C_fpx[i][j] = fxp_double_to_fxp(_controller.C[i][j]);
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < nInputs; j++)
    {
      D_fpx[i][j] = fxp_double_to_fxp(_controller.D[i][j]);
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < 1; j++)
    {
      states_fpx[i][j] = fxp_double_to_fxp(_controller.states[i][j]);
    }
  }

  for(i = 0; i < nInputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      inputs_fpx[i][j] = fxp_double_to_fxp(_controller.inputs[i][j]);
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      outputs_fpx[i][j] = fxp_double_to_fxp(_controller.outputs[i][j]);
    }
  }

  fxp_matrix_multiplication(
    nOutputs, nStates, nStates, 1, C_fpx, states_fpx, result1);
  fxp_matrix_multiplication(
    nOutputs, nInputs, nInputs, 1, D_fpx, inputs_fpx, result2);

  fxp_add_matrix(nOutputs, 1, result1, result2, outputs_fpx);

  for(i = 1; i < 7; i++)
  {
    fxp_matrix_multiplication(
      nStates, nStates, nStates, 1, A_fpx, states_fpx, result1);
    fxp_matrix_multiplication(
      nStates, nInputs, nInputs, 1, B_fpx, inputs_fpx, result2);

    fxp_add_matrix(nStates, 1, result1, result2, states_fpx);

    fxp_matrix_multiplication(
      nOutputs, nStates, nStates, 1, C_fpx, states_fpx, result1);
    fxp_matrix_multiplication(
      nOutputs, nInputs, nInputs, 1, D_fpx, inputs_fpx, result2);

    fxp_add_matrix(nOutputs, 1, result1, result2, outputs_fpx);
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < 1; j++)
    {
      _controller.states[i][j] = fxp_to_double(states_fpx[i][j]);
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      _controller.outputs[i][j] = fxp_to_double(outputs_fpx[i][j]);
    }
  }

  return _controller.outputs[0][0];
}
# 28 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2

# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_overflow.h" 1
# 17 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_overflow.h"
int nondet_int();
float nondet_float();

extern digital_system ds;
extern implementation impl;

int verify_overflow(void)
{
# 71 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_overflow.h"
  fxp_t min_fxp = fxp_double_to_fxp(impl.min);
  fxp_t max_fxp = fxp_double_to_fxp(impl.max);

  fxp_t y[X_SIZE_VALUE];
  fxp_t x[X_SIZE_VALUE];

  int i;
  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
    y[i] = 0;
    x[i] = nondet_int();
    __DSVERIFIER_assume(x[i] >= min_fxp && x[i] <= max_fxp);
  }

  int Nw = 0;

  Nw = ds.a_size > ds.b_size ? ds.a_size : ds.b_size;

  fxp_t yaux[ds.a_size];
  fxp_t xaux[ds.b_size];
  fxp_t waux[Nw];

  for(i = 0; i < ds.a_size; ++i)
  {
    yaux[i] = 0;
  }
  for(i = 0; i < ds.b_size; ++i)
  {
    xaux[i] = 0;
  }
  for(i = 0; i < Nw; ++i)
  {
    waux[i] = 0;
  }

  fxp_t xk, temp;
  fxp_t *aptr, *bptr, *xptr, *yptr, *wptr;

  int j;
  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
# 172 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_overflow.h"
  }

  overflow_mode = 1;
  fxp_verify_overflow_array(y, X_SIZE_VALUE);

  return 0;
}
# 30 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_limit_cycle.h" 1
# 13 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_limit_cycle.h"
extern digital_system ds;
extern implementation impl;
extern digital_system_state_space _controller;

extern int nStates;
extern int nInputs;
extern int nOutputs;

int verify_limit_cycle_state_space(void)
{
  double stateMatrix[4][4];
  double outputMatrix[4][4];
  double arrayLimitCycle[4];

  double result1[4][4];
  double result2[4][4];

  int i, j, k;

  for(i = 0; i < 4; i++)
  {
    for(j = 0; j < 4; j++)
    {
      result1[i][j] = 0;
      result2[i][j] = 0;
      stateMatrix[i][j] = 0;
      outputMatrix[i][j] = 0;
    }
  }

  double_matrix_multiplication(
    nOutputs, nStates, nStates, 1, _controller.C, _controller.states, result1);
  double_matrix_multiplication(
    nOutputs, nInputs, nInputs, 1, _controller.D, _controller.inputs, result2);

  double_add_matrix(nOutputs, 1, result1, result2, _controller.outputs);

  k = 0;

  for(i = 1; i < 7; i++)
  {
    double_matrix_multiplication(
      nStates, nStates, nStates, 1, _controller.A, _controller.states, result1);
    double_matrix_multiplication(
      nStates, nInputs, nInputs, 1, _controller.B, _controller.inputs, result2);

    double_add_matrix(nStates, 1, result1, result2, _controller.states);

    double_matrix_multiplication(
      nOutputs,
      nStates,
      nStates,
      1,
      _controller.C,
      _controller.states,
      result1);
    double_matrix_multiplication(
      nOutputs,
      nInputs,
      nInputs,
      1,
      _controller.D,
      _controller.inputs,
      result2);

    double_add_matrix(nOutputs, 1, result1, result2, _controller.outputs);

    int l;
    for(l = 0; l < nStates; l++)
    {
      stateMatrix[l][k] = _controller.states[l][0];
    }
    for(l = 0; l < nOutputs; l++)
    {
      stateMatrix[l][k] = _controller.outputs[l][0];
    }
    k++;
  }

  printf("#matrix STATES -------------------------------");
  print_matrix(stateMatrix, nStates, 7);

  printf("#matrix OUTPUTS -------------------------------");
  print_matrix(outputMatrix, nOutputs, 7);
  ((0) ? (void)0
       : _assert(
           "0",
           "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/"
           "verify_limit_cycle.h",
           91));

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < 7; j++)
    {
      arrayLimitCycle[j] = stateMatrix[i][j];
    }
    double_check_persistent_limit_cycle(arrayLimitCycle, 7);
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < 7; j++)
    {
      arrayLimitCycle[j] = outputMatrix[i][j];
    }
    double_check_persistent_limit_cycle(arrayLimitCycle, 7);
  }

  ((0) ? (void)0
       : _assert(
           "0",
           "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/"
           "verify_limit_cycle.h",
           108));
}

int verify_limit_cycle(void)
{
  overflow_mode = 3;

  int i;
  int Set_xsize_at_least_two_times_Na = 2 * ds.a_size;
  printf("X_SIZE must be at least 2 * ds.a_size");
  __DSVERIFIER_assert(X_SIZE_VALUE >= Set_xsize_at_least_two_times_Na);
# 166 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_limit_cycle.h"
  fxp_t y[X_SIZE_VALUE];
  fxp_t x[X_SIZE_VALUE];

  fxp_t min_fxp = fxp_double_to_fxp(impl.min);
  fxp_t max_fxp = fxp_double_to_fxp(impl.max);

  fxp_t xaux[ds.b_size];
  int nondet_constant_input = nondet_int();
  __DSVERIFIER_assume(
    nondet_constant_input >= min_fxp && nondet_constant_input <= max_fxp);
  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
    x[i] = nondet_constant_input;
    y[i] = 0;
  }
  for(i = 0; i < ds.b_size; ++i)
  {
    xaux[i] = nondet_constant_input;
  }

  int Nw = 0;

  Nw = ds.a_size > ds.b_size ? ds.a_size : ds.b_size;

  fxp_t yaux[ds.a_size];
  fxp_t y0[ds.a_size];

  fxp_t waux[Nw];
  fxp_t w0[Nw];
# 204 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_limit_cycle.h"
  for(i = 0; i < Nw; ++i)
  {
    waux[i] = nondet_int();
    __DSVERIFIER_assume(waux[i] >= min_fxp && waux[i] <= max_fxp);
    w0[i] = waux[i];
  }

  fxp_t xk, temp;
  fxp_t *aptr, *bptr, *xptr, *yptr, *wptr;

  int j;
  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
# 276 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_limit_cycle.h"
  }

  fxp_check_persistent_limit_cycle(y, X_SIZE_VALUE);

  return 0;
}
# 31 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_error.h" 1
# 17 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_error.h"
extern digital_system ds;
extern implementation impl;

int verify_error(void)
{
  overflow_mode = 2;

  double a_cascade[100];
  int a_cascade_size;
  double b_cascade[100];
  int b_cascade_size;
# 69 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_error.h"
  fxp_t min_fxp = fxp_double_to_fxp(impl.min);
  fxp_t max_fxp = fxp_double_to_fxp(impl.max);

  fxp_t y[X_SIZE_VALUE];
  fxp_t x[X_SIZE_VALUE];
  double yf[X_SIZE_VALUE];
  double xf[X_SIZE_VALUE];

  int Nw = 0;

  Nw = ds.a_size > ds.b_size ? ds.a_size : ds.b_size;

  fxp_t yaux[ds.a_size];
  fxp_t xaux[ds.b_size];
  fxp_t waux[Nw];

  double yfaux[ds.a_size];
  double xfaux[ds.b_size];
  double wfaux[Nw];

  int i;
  for(i = 0; i < ds.a_size; ++i)
  {
    yaux[i] = 0;
    yfaux[i] = 0;
  }
  for(i = 0; i < ds.b_size; ++i)
  {
    xaux[i] = 0;
    xfaux[i] = 0;
  }
  for(i = 0; i < Nw; ++i)
  {
    waux[i] = 0;
    wfaux[i] = 0;
  }

  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
    y[i] = 0;
    x[i] = nondet_int();
    __DSVERIFIER_assume(x[i] >= min_fxp && x[i] <= max_fxp);
    yf[i] = 0.0f;
    xf[i] = fxp_to_double(x[i]);
  }

  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
# 169 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_error.h"
    double absolute_error = yf[i] - fxp_to_double(y[i]);

    __DSVERIFIER_assert(
      absolute_error < (impl.max_error) && absolute_error > (-impl.max_error));
  }
  return 0;
}
# 32 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_zero_input_limit_cycle.h" 1
# 13 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_zero_input_limit_cycle.h"
extern digital_system ds;
extern implementation impl;

int verify_zero_input_limit_cycle(void)
{
  overflow_mode = 3;

  int i, j;
  int Set_xsize_at_least_two_times_Na = 2 * ds.a_size;
  printf("X_SIZE must be at least 2 * ds.a_size");
  ((X_SIZE_VALUE >= Set_xsize_at_least_two_times_Na)
     ? (void)0
     : _assert(
         "X_SIZE_VALUE >= Set_xsize_at_least_two_times_Na",
         "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/"
         "verify_zero_input_limit_cycle.h",
         23));
# 71 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_zero_input_limit_cycle.h"
  fxp_t min_fxp = fxp_double_to_fxp(impl.min);
  fxp_t max_fxp = fxp_double_to_fxp(impl.max);

  fxp_t y[X_SIZE_VALUE];
  fxp_t x[X_SIZE_VALUE];

  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
    y[i] = 0;
    x[i] = 0;
  }

  int Nw = 0;

  Nw = ds.a_size > ds.b_size ? ds.a_size : ds.b_size;

  fxp_t yaux[ds.a_size];
  fxp_t xaux[ds.b_size];
  fxp_t waux[Nw];

  fxp_t y0[ds.a_size];
  fxp_t w0[Nw];
# 104 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_zero_input_limit_cycle.h"
  for(i = 0; i < Nw; ++i)
  {
    waux[i] = nondet_int();
    __DSVERIFIER_assume(waux[i] >= min_fxp && waux[i] <= max_fxp);
    w0[i] = waux[i];
  }

  for(i = 0; i < ds.b_size; ++i)
  {
    xaux[i] = 0;
  }

  fxp_t xk, temp;
  fxp_t *aptr, *bptr, *xptr, *yptr, *wptr;

  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
# 188 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_zero_input_limit_cycle.h"
  }

  fxp_check_persistent_limit_cycle(y, X_SIZE_VALUE);

  return 0;
}
# 33 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_generic_timing.h" 1
# 16 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_generic_timing.h"
int nondet_int();
float nondet_float();

extern digital_system ds;
extern implementation impl;
extern hardware hw;

int generic_timer = 0;

int verify_generic_timing(void)
{
  double y[X_SIZE_VALUE];
  double x[X_SIZE_VALUE];
  int i;
  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
    y[i] = 0;
    x[i] = nondet_float();
    __DSVERIFIER_assume(x[i] >= impl.min && x[i] <= impl.max);
  }

  int Nw = 0;

  Nw = ds.a_size > ds.b_size ? ds.a_size : ds.b_size;

  double yaux[ds.a_size];
  double xaux[ds.b_size];
  double waux[Nw];

  for(i = 0; i < ds.a_size; ++i)
  {
    yaux[i] = 0;
  }
  for(i = 0; i < ds.b_size; ++i)
  {
    xaux[i] = 0;
  }
  for(i = 0; i < Nw; ++i)
  {
    waux[i] = 0;
  }

  double xk, temp;
  double *aptr, *bptr, *xptr, *yptr, *wptr;

  int j;

  generic_timer += ((2 * hw.assembly.std) + (1 * hw.assembly.rjmp));
  double initial_timer = generic_timer;
  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
    generic_timer +=
      ((2 * hw.assembly.ldd) + (1 * hw.assembly.adiw) + (2 * hw.assembly.std));
    generic_timer +=
      ((2 * hw.assembly.ldd) + (1 * hw.assembly.cpi) + (1 * hw.assembly.cpc) +
       (1 * hw.assembly.brlt));
# 88 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_generic_timing.h"
    double spent_time = (((double)generic_timer) * hw.cycle);
    ((spent_time <= ds.sample_time)
       ? (void)0
       : _assert(
           "spent_time <= ds.sample_time",
           "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/"
           "verify_generic_timing.h",
           89));
    generic_timer = initial_timer;
  }
  return 0;
}
# 34 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_timing_msp430.h" 1
# 16 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_timing_msp430.h"
int nondet_int();
float nondet_float();

extern digital_system ds;
extern implementation impl;

int verify_timing_msp_430(void)
{
  double y[X_SIZE_VALUE];
  double x[X_SIZE_VALUE];
  int i;
  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
    y[i] = 0;
    x[i] = nondet_float();
    __DSVERIFIER_assume(x[i] >= impl.min && x[i] <= impl.max);
  }

  int Nw = 0;

  Nw = ds.a_size > ds.b_size ? ds.a_size : ds.b_size;

  double yaux[ds.a_size];
  double xaux[ds.b_size];
  double waux[Nw];

  for(i = 0; i < ds.a_size; ++i)
  {
    yaux[i] = 0;
  }
  for(i = 0; i < ds.b_size; ++i)
  {
    xaux[i] = 0;
  }
  for(i = 0; i < Nw; ++i)
  {
    waux[i] = 0;
  }

  double xk, temp;
  double *aptr, *bptr, *xptr, *yptr, *wptr;

  int j;
  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
# 121 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_timing_msp430.h"
  }
  return 0;
}
# 35 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_stability.h" 1
# 21 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_stability.h"
extern digital_system ds;
extern implementation impl;

int verify_stability(void)
{
  overflow_mode = 0;
# 83 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_stability.h"
  return 0;
}
# 36 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_minimum_phase.h" 1
# 21 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_minimum_phase.h"
extern digital_system ds;
extern implementation impl;

int verify_minimum_phase(void)
{
  overflow_mode = 0;
# 85 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_minimum_phase.h"
  return 0;
}
# 37 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_stability_closedloop.h" 1
# 17 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_stability_closedloop.h"
extern digital_system plant;
extern digital_system plant_cbmc;
extern digital_system controller;

int verify_stability_closedloop_using_dslib(void)
{
  double *c_num = controller.b;
  int c_num_size = controller.b_size;
  double *c_den = controller.a;
  int c_den_size = controller.a_size;

  fxp_t c_num_fxp[controller.b_size];
  fxp_double_to_fxp_array(c_num, c_num_fxp, controller.b_size);
  fxp_t c_den_fxp[controller.a_size];
  fxp_double_to_fxp_array(c_den, c_den_fxp, controller.a_size);

  double c_num_qtz[controller.b_size];
  fxp_to_double_array(c_num_qtz, c_num_fxp, controller.b_size);
  double c_den_qtz[controller.a_size];
  fxp_to_double_array(c_den_qtz, c_den_fxp, controller.a_size);
# 48 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_stability_closedloop.h"
  double *p_num = plant_cbmc.b;
  int p_num_size = plant.b_size;
  double *p_den = plant_cbmc.a;
  int p_den_size = plant.a_size;

  double ans_num[100];
  int ans_num_size = controller.b_size + plant.b_size - 1;
  double ans_den[100];
  int ans_den_size = controller.a_size + plant.a_size - 1;
# 68 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_stability_closedloop.h"
  printf("Verifying stability for closedloop function\n");
  __DSVERIFIER_assert(check_stability_closedloop(
    ans_den, ans_den_size, p_num, p_num_size, p_den, p_den_size));

  return 0;
}
# 38 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_limit_cycle_closedloop.h" 1
# 23 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_limit_cycle_closedloop.h"
extern digital_system plant;
extern digital_system plant_cbmc;
extern digital_system controller;

double nondet_double();

int verify_limit_cycle_closed_loop(void)
{
  overflow_mode = 3;

  double *c_num = controller.b;
  int c_num_size = controller.b_size;
  double *c_den = controller.a;
  int c_den_size = controller.a_size;

  fxp_t c_num_fxp[controller.b_size];
  fxp_double_to_fxp_array(c_num, c_num_fxp, controller.b_size);
  fxp_t c_den_fxp[controller.a_size];
  fxp_double_to_fxp_array(c_den, c_den_fxp, controller.a_size);

  double c_num_qtz[controller.b_size];
  fxp_to_double_array(c_num_qtz, c_num_fxp, controller.b_size);
  double c_den_qtz[controller.a_size];
  fxp_to_double_array(c_den_qtz, c_den_fxp, controller.a_size);
# 58 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_limit_cycle_closedloop.h"
  double *p_num = plant_cbmc.b;
  int p_num_size = plant.b_size;
  double *p_den = plant_cbmc.a;
  int p_den_size = plant.a_size;

  double ans_num[100];
  int ans_num_size = controller.b_size + plant.b_size - 1;
  double ans_den[100];
  int ans_den_size = controller.a_size + plant.a_size - 1;

  int i;
  double y[X_SIZE_VALUE];
  double x[X_SIZE_VALUE];

  double xaux[ans_num_size];
  double nondet_constant_input = nondet_double();
  __DSVERIFIER_assume(
    nondet_constant_input >= impl.min && nondet_constant_input <= impl.max);
  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
    x[i] = nondet_constant_input;
    y[i] = 0;
  }
  for(i = 0; i < ans_num_size; ++i)
  {
    xaux[i] = nondet_constant_input;
  }

  double yaux[ans_den_size];
  double y0[ans_den_size];

  int Nw = ans_den_size > ans_num_size ? ans_den_size : ans_num_size;
  double waux[Nw];
  double w0[Nw];
# 105 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_limit_cycle_closedloop.h"
  for(i = 0; i < Nw; ++i)
  {
    waux[i] = nondet_int();
    __DSVERIFIER_assume(waux[i] >= impl.min && waux[i] <= impl.max);
    w0[i] = waux[i];
  }

  double xk, temp;
  double *aptr, *bptr, *xptr, *yptr, *wptr;

  int j;
  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
# 137 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_limit_cycle_closedloop.h"
  }

  double_check_persistent_limit_cycle(y, X_SIZE_VALUE);

  return 0;
}
# 39 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_error_closedloop.h" 1
# 23 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_error_closedloop.h"
extern digital_system plant;
extern digital_system plant_cbmc;
extern digital_system controller;

int verify_error_closedloop(void)
{
  overflow_mode = 3;

  double *c_num = controller.b;
  int c_num_size = controller.b_size;
  double *c_den = controller.a;
  int c_den_size = controller.a_size;

  fxp_t c_num_fxp[controller.b_size];
  fxp_double_to_fxp_array(c_num, c_num_fxp, controller.b_size);
  fxp_t c_den_fxp[controller.a_size];
  fxp_double_to_fxp_array(c_den, c_den_fxp, controller.a_size);

  double c_num_qtz[controller.b_size];
  fxp_to_double_array(c_num_qtz, c_num_fxp, controller.b_size);
  double c_den_qtz[controller.a_size];
  fxp_to_double_array(c_den_qtz, c_den_fxp, controller.a_size);
# 56 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_error_closedloop.h"
  double *p_num = plant_cbmc.b;
  int p_num_size = plant.b_size;
  double *p_den = plant_cbmc.a;
  int p_den_size = plant.a_size;

  double ans_num_double[100];
  double ans_num_qtz[100];
  int ans_num_size = controller.b_size + plant.b_size - 1;
  double ans_den_qtz[100];
  double ans_den_double[100];
  int ans_den_size = controller.a_size + plant.a_size - 1;
# 77 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_error_closedloop.h"
  int i;
  double y_qtz[X_SIZE_VALUE];
  double y_double[X_SIZE_VALUE];
  double x_qtz[X_SIZE_VALUE];
  double x_double[X_SIZE_VALUE];
  double xaux_qtz[ans_num_size];
  double xaux_double[ans_num_size];

  double xaux[ans_num_size];
  double nondet_constant_input = nondet_double();
  __DSVERIFIER_assume(
    nondet_constant_input >= impl.min && nondet_constant_input <= impl.max);
  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
    x_qtz[i] = nondet_constant_input;
    x_double[i] = nondet_constant_input;
    y_qtz[i] = 0;
    y_double[i] = 0;
  }
  for(i = 0; i < ans_num_size; ++i)
  {
    xaux_qtz[i] = nondet_constant_input;
    xaux_double[i] = nondet_constant_input;
  }

  double yaux_qtz[ans_den_size];
  double yaux_double[ans_den_size];
  double y0_qtz[ans_den_size];
  double y0_double[ans_den_size];

  int Nw = ans_den_size > ans_num_size ? ans_den_size : ans_num_size;
  double waux_qtz[Nw];
  double waux_double[Nw];
  double w0_qtz[Nw];
  double w0_double[Nw];

  for(i = 0; i < Nw; ++i)
  {
    waux_qtz[i] = 0;
    waux_double[i] = 0;
  }

  for(i = 0; i < X_SIZE_VALUE; ++i)
  {
# 156 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_error_closedloop.h"
    double absolute_error = y_double[i] - fxp_to_double(y_qtz[i]);

    __DSVERIFIER_assert(
      absolute_error < (impl.max_error) && absolute_error > (-impl.max_error));
  }

  return 0;
}
# 40 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_error_state_space.h" 1
# 17 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_error_state_space.h"
extern digital_system_state_space _controller;
extern double error_limit;
extern int closed_loop;

double ss_system_quantization_error()
{
  digital_system_state_space __backupController;
  int i;
  int j;

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < nStates; j++)
    {
      __backupController.A[i][j] = (_controller.A[i][j]);
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < nInputs; j++)
    {
      __backupController.B[i][j] = (_controller.B[i][j]);
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < nStates; j++)
    {
      __backupController.C[i][j] = (_controller.C[i][j]);
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < nInputs; j++)
    {
      __backupController.D[i][j] = (_controller.D[i][j]);
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < 1; j++)
    {
      __backupController.states[i][j] = (_controller.states[i][j]);
    }
  }

  for(i = 0; i < nInputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      __backupController.inputs[i][j] = (_controller.inputs[i][j]);
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      __backupController.outputs[i][j] = (_controller.outputs[i][j]);
    }
  }

  double __quant_error = 0.0;

  double output_double = double_state_space_representation();

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < nStates; j++)
    {
      _controller.A[i][j] = __backupController.A[i][j];
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < nInputs; j++)
    {
      _controller.B[i][j] = __backupController.B[i][j];
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < nStates; j++)
    {
      _controller.C[i][j] = __backupController.C[i][j];
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < nInputs; j++)
    {
      _controller.D[i][j] = __backupController.D[i][j];
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < 1; j++)
    {
      _controller.states[i][j] = __backupController.states[i][j];
    }
  }

  for(i = 0; i < nInputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      _controller.inputs[i][j] = __backupController.inputs[i][j];
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      _controller.outputs[i][j] = __backupController.outputs[i][j];
    }
  }

  double output_fxp = fxp_state_space_representation();

  fxp_verify_overflow(output_fxp);

  __quant_error = output_double - fxp_to_double(output_fxp);

  return __quant_error;
}

double ss_closed_loop_quantization_error()
{
  double reference[4][4];
  double result1[4][4];
  double result2[4][4];
  unsigned int i;
  unsigned int j;
  short unsigned int flag = 0;

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < nInputs; j++)
    {
      if(_controller.D[i][j] != 0)
      {
        flag = 1;
      }
    }
  }

  for(i = 0; i < nInputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      reference[i][j] = (_controller.inputs[i][j]);
    }
  }

  for(i = 0; i < 4; i++)
  {
    for(j = 0; j < 4; j++)
    {
      result1[i][j] = 0;
      result2[i][j] = 0;
    }
  }

  for(i = 1; i < 7; i++)
  {
    double_matrix_multiplication(
      nOutputs,
      nStates,
      nStates,
      1,
      _controller.C,
      _controller.states,
      result1);

    if(flag == 1)
    {
      double_matrix_multiplication(
        nOutputs,
        nInputs,
        nInputs,
        1,
        _controller.D,
        _controller.inputs,
        result2);
    }

    double_add_matrix(nOutputs, 1, result1, result2, _controller.outputs);

    double_matrix_multiplication(
      nInputs,
      nOutputs,
      nOutputs,
      1,
      _controller.K,
      _controller.outputs,
      result1);

    printf("### U (before) = %.9f", _controller.inputs[0][0]);
    printf("### reference = %.9f", reference[0][0]);
    printf("### result1 = %.9f", result1[0][0]);
    printf("### reference - result1 = %.9f", (reference[0][0] - result1[0][0]));

    double_sub_matrix(nInputs, 1, reference, result1, _controller.inputs);

    printf("### Y = %.9f", _controller.outputs[0][0]);
    printf("### U (after) = %.9f \n### \n### ", _controller.inputs[0][0]);

    double_matrix_multiplication(
      nStates, nStates, nStates, 1, _controller.A, _controller.states, result1);
    double_matrix_multiplication(
      nStates, nInputs, nInputs, 1, _controller.B, _controller.inputs, result2);

    double_add_matrix(nStates, 1, result1, result2, _controller.states);
  }

  return _controller.outputs[0][0];
}

double fxp_ss_closed_loop_quantization_error()
{
  double reference[4][4];
  double result1[4][4];
  double result2[4][4];
  fxp_t K_fpx[4][4];
  fxp_t outputs_fpx[4][4];
  fxp_t result_fxp[4][4];
  unsigned int i;
  unsigned int j;
  unsigned int k;
  short unsigned int flag = 0;

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < nInputs; j++)
    {
      if(_controller.D[i][j] != 0)
      {
        flag = 1;
      }
    }
  }

  for(i = 0; i < nInputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      reference[i][j] = (_controller.inputs[i][j]);
    }
  }

  for(i = 0; i < nInputs; i++)
  {
    for(j = 0; j < nOutputs; j++)
    {
      K_fpx[i][j] = 0;
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      outputs_fpx[i][j] = 0;
    }
  }

  for(i = 0; i < 4; i++)
  {
    for(j = 0; j < 4; j++)
    {
      result_fxp[i][j] = 0;
    }
  }

  for(i = 0; i < nInputs; i++)
  {
    for(j = 0; j < nOutputs; j++)
    {
      K_fpx[i][j] = fxp_double_to_fxp(_controller.K[i][j]);
    }
  }

  for(i = 0; i < 4; i++)
  {
    for(j = 0; j < 4; j++)
    {
      result1[i][j] = 0;
      result2[i][j] = 0;
    }
  }

  for(i = 1; i < 7; i++)
  {
    double_matrix_multiplication(
      nOutputs,
      nStates,
      nStates,
      1,
      _controller.C,
      _controller.states,
      result1);

    if(flag == 1)
    {
      double_matrix_multiplication(
        nOutputs,
        nInputs,
        nInputs,
        1,
        _controller.D,
        _controller.inputs,
        result2);
    }

    double_add_matrix(nOutputs, 1, result1, result2, _controller.outputs);

    for(k = 0; k < nOutputs; k++)
    {
      for(j = 0; j < 1; j++)
      {
        outputs_fpx[k][j] = fxp_double_to_fxp(_controller.outputs[k][j]);
      }
    }

    fxp_matrix_multiplication(
      nInputs, nOutputs, nOutputs, 1, K_fpx, outputs_fpx, result_fxp);

    for(k = 0; k < nInputs; k++)
    {
      for(j = 0; j < 1; j++)
      {
        result1[k][j] = fxp_to_double(result_fxp[k][j]);
      }
    }

    printf("### fxp: U (before) = %.9f", _controller.inputs[0][0]);
    printf("### fxp: reference = %.9f", reference[0][0]);
    printf("### fxp: result1 = %.9f", result1[0][0]);
    printf(
      "### fxp: reference - result1 = %.9f", (reference[0][0] - result1[0][0]));

    double_sub_matrix(nInputs, 1, reference, result1, _controller.inputs);

    printf("### fxp: Y = %.9f", _controller.outputs[0][0]);
    printf("### fxp: U (after) = %.9f \n### \n### ", _controller.inputs[0][0]);

    double_matrix_multiplication(
      nStates, nStates, nStates, 1, _controller.A, _controller.states, result1);
    double_matrix_multiplication(
      nStates, nInputs, nInputs, 1, _controller.B, _controller.inputs, result2);

    double_add_matrix(nStates, 1, result1, result2, _controller.states);
  }

  return _controller.outputs[0][0];
}

int verify_error_state_space(void)
{
  overflow_mode = 0;

  double __quant_error;

  if(closed_loop)
  {
    __quant_error = ss_closed_loop_quantization_error() -
                    fxp_ss_closed_loop_quantization_error();
  }
  else
  {
    __quant_error = ss_system_quantization_error();
  }

  ((__quant_error < error_limit && __quant_error > (-error_limit))
     ? (void)0
     : _assert(
         "__quant_error < error_limit && __quant_error > (-error_limit)",
         "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/"
         "verify_error_state_space.h",
         323));

  return 0;
}
# 41 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_controllability.h" 1
# 14 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_controllability.h"
extern digital_system_state_space _controller;

int verify_controllability(void)
{
  int i;
  int j;

  fxp_t A_fpx[4][4];
  fxp_t B_fpx[4][4];
  fxp_t controllabilityMatrix[4][4];
  fxp_t backup[4][4];
  fxp_t backupSecond[4][4];
  double controllabilityMatrix_double[4][4];

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < (nStates * nInputs); j++)
    {
      A_fpx[i][j] = 0.0;
      B_fpx[i][j] = 0.0;
      controllabilityMatrix[i][j] = 0.0;
      backup[i][j] = 0.0;
      backupSecond[i][j] = 0.0;
      controllabilityMatrix_double[i][j] = 0.0;
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < nStates; j++)
    {
      A_fpx[i][j] = fxp_double_to_fxp(_controller.A[i][j]);
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < nInputs; j++)
    {
      B_fpx[i][j] = fxp_double_to_fxp(_controller.B[i][j]);
    }
  }

  if(nInputs > 1)
  {
    int l = 0;

    for(j = 0; j < (nStates * nInputs);)
    {
      fxp_exp_matrix(nStates, nStates, A_fpx, l, backup);
      l++;
      fxp_matrix_multiplication(
        nStates, nStates, nStates, nInputs, backup, B_fpx, backupSecond);
      for(int k = 0; k < nInputs; k++)
      {
        for(i = 0; i < nStates; i++)
        {
          controllabilityMatrix[i][j] = backupSecond[i][k];
        }
        j++;
      }
    }

    for(i = 0; i < nStates; i++)
    {
      for(j = 0; j < (nStates * nInputs); j++)
      {
        backup[i][j] = 0.0;
      }
    }

    fxp_transpose(controllabilityMatrix, backup, nStates, (nStates * nInputs));

    fxp_t mimo_controllabilityMatrix_fxp[4][4];
    fxp_matrix_multiplication(
      nStates,
      (nStates * nInputs),
      (nStates * nInputs),
      nStates,
      controllabilityMatrix,
      backup,
      mimo_controllabilityMatrix_fxp);

    for(i = 0; i < nStates; i++)
    {
      for(j = 0; j < nStates; j++)
      {
        controllabilityMatrix_double[i][j] =
          fxp_to_double(mimo_controllabilityMatrix_fxp[i][j]);
      }
    }

    ((determinant(controllabilityMatrix_double, nStates) != 0)
       ? (void)0
       : _assert(
           "determinant(controllabilityMatrix_double,nStates) != 0",
           "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/"
           "verify_controllability.h",
           91));
  }
  else
  {
    for(j = 0; j < nStates; j++)
    {
      fxp_exp_matrix(nStates, nStates, A_fpx, j, backup);
      fxp_matrix_multiplication(
        nStates, nStates, nStates, nInputs, backup, B_fpx, backupSecond);
      for(i = 0; i < nStates; i++)
      {
        controllabilityMatrix[i][j] = backupSecond[i][0];
      }
    }

    for(i = 0; i < nStates; i++)
    {
      for(j = 0; j < nStates; j++)
      {
        controllabilityMatrix_double[i][j] =
          fxp_to_double(controllabilityMatrix[i][j]);
      }
    }

    ((determinant(controllabilityMatrix_double, nStates) != 0)
       ? (void)0
       : _assert(
           "determinant(controllabilityMatrix_double,nStates) != 0",
           "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/"
           "verify_controllability.h",
           113));
  }

  return 0;
}

int verify_controllability_double(void)
{
  int i;
  int j;

  double controllabilityMatrix[4][4];
  double backup[4][4];
  double backupSecond[4][4];
  double controllabilityMatrix_double[4][4];

  if(nInputs > 1)
  {
    int l = 0;
    for(j = 0; j < (nStates * nInputs);)
    {
      double_exp_matrix(nStates, nStates, _controller.A, l, backup);
      l++;
      double_matrix_multiplication(
        nStates,
        nStates,
        nStates,
        nInputs,
        backup,
        _controller.B,
        backupSecond);
      for(int k = 0; k < nInputs; k++)
      {
        for(i = 0; i < nStates; i++)
        {
          controllabilityMatrix[i][j] = backupSecond[i][k];
        }
        j++;
      }
    }

    for(i = 0; i < nStates; i++)
    {
      for(j = 0; j < (nStates * nInputs); j++)
      {
        backup[i][j] = 0.0;
      }
    }

    transpose(controllabilityMatrix, backup, nStates, (nStates * nInputs));

    double mimo_controllabilityMatrix_double[4][4];
    double_matrix_multiplication(
      nStates,
      (nStates * nInputs),
      (nStates * nInputs),
      nStates,
      controllabilityMatrix,
      backup,
      mimo_controllabilityMatrix_double);
    ((determinant(mimo_controllabilityMatrix_double, nStates) != 0)
       ? (void)0
       : _assert(
           "determinant(mimo_controllabilityMatrix_double,nStates) != 0",
           "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/"
           "verify_controllability.h",
           154));
  }
  else
  {
    for(j = 0; j < nStates; j++)
    {
      double_exp_matrix(nStates, nStates, _controller.A, j, backup);
      double_matrix_multiplication(
        nStates,
        nStates,
        nStates,
        nInputs,
        backup,
        _controller.B,
        backupSecond);
      for(i = 0; i < nStates; i++)
      {
        controllabilityMatrix[i][j] = backupSecond[i][0];
      }
    }
    ((determinant(controllabilityMatrix, nStates) != 0)
       ? (void)0
       : _assert(
           "determinant(controllabilityMatrix,nStates) != 0",
           "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/"
           "verify_controllability.h",
           163));
  }

  return 0;
}
# 42 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2
# 1 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_observability.h" 1
# 17 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_observability.h"
extern digital_system_state_space _controller;

int verify_observability(void)
{
  int i;
  int j;

  fxp_t A_fpx[4][4];
  fxp_t C_fpx[4][4];
  fxp_t observabilityMatrix[4][4];
  fxp_t backup[4][4];
  fxp_t backupSecond[4][4];
  double observabilityMatrix_double[4][4];

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < nStates; j++)
    {
      observabilityMatrix[i][j] = 0;
      A_fpx[i][j] = 0;
      C_fpx[i][j] = 0;
      backup[i][j] = 0;
      backupSecond[i][j] = 0;
    }
  }

  for(i = 0; i < nStates; i++)
  {
    for(j = 0; j < nStates; j++)
    {
      A_fpx[i][j] = fxp_double_to_fxp(_controller.A[i][j]);
    }
  }

  for(i = 0; i < nOutputs; i++)
  {
    for(j = 0; j < nStates; j++)
    {
      C_fpx[i][j] = fxp_double_to_fxp(_controller.C[i][j]);
    }
  }

  if(nOutputs > 1)
  {
    int l;
    j = 0;
    for(l = 0; l < nStates;)
    {
      fxp_exp_matrix(nStates, nStates, A_fpx, l, backup);
      l++;
      fxp_matrix_multiplication(
        nOutputs, nStates, nStates, nStates, C_fpx, backup, backupSecond);
      for(int k = 0; k < nOutputs; k++)
      {
        for(i = 0; i < nStates; i++)
        {
          observabilityMatrix[j][i] = backupSecond[k][i];
        }
        j++;
      }
    }
# 80 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_observability.h"
    for(i = 0; i < nStates; i++)
    {
      for(j = 0; j < (nStates * nOutputs); j++)
      {
        backup[i][j] = 0.0;
      }
    }

    fxp_transpose(observabilityMatrix, backup, (nStates * nOutputs), nStates);
# 99 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_observability.h"
    fxp_t mimo_observabilityMatrix_fxp[4][4];
    fxp_matrix_multiplication(
      nStates,
      (nStates * nOutputs),
      (nStates * nOutputs),
      nStates,
      backup,
      observabilityMatrix,
      mimo_observabilityMatrix_fxp);
# 112 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/verify_observability.h"
    for(i = 0; i < nStates; i++)
    {
      for(j = 0; j < nStates; j++)
      {
        observabilityMatrix_double[i][j] =
          fxp_to_double(mimo_observabilityMatrix_fxp[i][j]);
      }
    }

    ((determinant(observabilityMatrix_double, nStates) != 0)
       ? (void)0
       : _assert(
           "determinant(observabilityMatrix_double,nStates) != 0",
           "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/"
           "verify_observability.h",
           119));
  }
  else
  {
    for(i = 0; i < nStates; i++)
    {
      fxp_exp_matrix(nStates, nStates, A_fpx, i, backup);
      fxp_matrix_multiplication(
        nOutputs, nStates, nStates, nStates, C_fpx, backup, backupSecond);
      for(j = 0; j < nStates; j++)
      {
        observabilityMatrix[i][j] = backupSecond[0][j];
      }
    }

    for(i = 0; i < nStates; i++)
    {
      for(j = 0; j < nStates; j++)
      {
        observabilityMatrix_double[i][j] =
          fxp_to_double(observabilityMatrix[i][j]);
      }
    }
    ((determinant(observabilityMatrix_double, nStates) != 0)
       ? (void)0
       : _assert(
           "determinant(observabilityMatrix_double,nStates) != 0",
           "c:/Users/Pascal/Software/cpp/dsverifier/bmc/engine/"
           "verify_observability.h",
           134));
  }

  return 0;
}
# 43 "c:/Users/Pascal/Software/cpp/dsverifier/bmc/dsverifier.h" 2

extern digital_system ds;
extern digital_system plant;
digital_system plant_cbmc;
extern digital_system controller;
extern implementation impl;
extern hardware hw;
extern digital_system_state_space _controller;

extern void initials();

void validation();
void call_verification_task(void *verification_task);
void call_closedloop_verification_task(void *closedloop_verification_task);
float nondet_float();
double nondet_double();

int main()
{
  initialization();
  validation();

  if(1 == 0)
    rounding_mode = 0;
  else if(1 == 1)
    rounding_mode = 1;
  else if(1 == 2)
    rounding_mode = 2;

  if(12 == 3)
  {
    call_verification_task(&verify_overflow);
  }
  else if(12 == 2)
  {
    call_verification_task(&verify_limit_cycle);
  }
  else if(12 == 6)
  {
    call_verification_task(&verify_error);
  }
  else if(12 == 1)
  {
    call_verification_task(&verify_zero_input_limit_cycle);
  }
  else if(12 == 4)
  {
    call_verification_task(&verify_timing_msp_430);
  }
  else if(12 == 5)
  {
    call_verification_task(&verify_generic_timing);
  }
  else if(12 == 7)
  {
    call_verification_task(&verify_stability);
  }
  else if(12 == 8)
  {
    call_verification_task(&verify_minimum_phase);
  }
  else if(12 == 9)
  {
    call_closedloop_verification_task(&verify_stability_closedloop_using_dslib);
  }
  else if(12 == 10)
  {
    call_closedloop_verification_task(&verify_limit_cycle_closed_loop);
  }
  else if(12 == 11)
  {
    call_closedloop_verification_task(&verify_error_closedloop);
  }
  else if(12 == 12)
  {
    verify_error_state_space();
  }
  else if(12 == 13)
  {
    verify_controllability();
  }
  else if(12 == 14)
  {
    verify_observability();
  }
  else if(12 == 15)
  {
    verify_limit_cycle_state_space();
  }

  return 0;
}

void validation()
{
  if(12 == 12 || 12 == 15 || 12 == 13 || 12 == 14)
  {
    if(7 == 0)
    {
      printf(
        "\n\n******************************************************************"
        "**************************\n");
      printf(
        "* set a K_SIZE to use this property in DSVerifier (use: "
        "-DK_SIZE=VALUE) *\n");
      printf(
        "**********************************************************************"
        "**********************\n");
      __DSVERIFIER_assert(0);
      exit(1);
    }
    initials();
    return;
  }
  if(
    ((12 != 9) && (12 != 10) && (12 != 11)) &&
    (ds.a_size == 0 || ds.b_size == 0))
  {
    printf(
      "\n\n********************************************************************"
      "********\n");
    printf("* set (ds and impl) parameters to check with DSVerifier *\n");
    printf(
      "************************************************************************"
      "****\n");
    __DSVERIFIER_assert(0);
  }
  if((12 == 9) || (12 == 10) || (12 == 11))
  {
    if(controller.a_size == 0 || plant.b_size == 0 || impl.int_bits == 0)
    {
      printf(
        "\n\n******************************************************************"
        "***********************************\n");
      printf(
        "* set (controller, plant, and impl) parameters to check CLOSED LOOP "
        "with DSVerifier *\n");
      printf(
        "**********************************************************************"
        "*******************************\n");
      __DSVERIFIER_assert(0);
    }
    else
    {
      printf(
        "\n\n******************************************************************"
        "***********************************\n");
      printf(
        "* set (controller and impl) parameters so that they do not overflow "
        "*\n");
      printf(
        "**********************************************************************"
        "*******************************\n");
      unsigned j;
      for(j = 0; j < controller.a_size; ++j)
      {
        const double value = controller.a[j];
        __DSVERIFIER_assert(value <= _dbl_max);
        __DSVERIFIER_assert(value >= _dbl_min);
      }
      for(j = 0; j < controller.b_size; ++j)
      {
        const double value = controller.b[j];
        __DSVERIFIER_assert(value <= _dbl_max);
        __DSVERIFIER_assert(value >= _dbl_min);
      }
    }

    if(controller.b_size > 0)
    {
      unsigned j, zeros = 0;
      for(j = 0; j < controller.b_size; ++j)
      {
        if(controller.b[j] == 0)
          ++zeros;
      }

      if(zeros == controller.b_size)
      {
        printf(
          "\n\n****************************************************************"
          "*************************************\n");
        printf("* The controller numerator must not be zero *\n");
        printf(
          "********************************************************************"
          "*********************************\n");
        __DSVERIFIER_assert(0);
      }
    }
    if(controller.a_size > 0)
    {
      unsigned j, zeros = 0;
      for(j = 0; j < controller.a_size; ++j)
      {
        if(controller.a[j] == 0)
          ++zeros;
      }
      if(zeros == controller.a_size)
      {
        printf(
          "\n\n****************************************************************"
          "*************************************\n");
        printf("* The controller denominator must not be zero *\n");
        printf(
          "********************************************************************"
          "*********************************\n");
        __DSVERIFIER_assert(0);
      }
    }

    if(0 == 0)
    {
      printf(
        "\n\n******************************************************************"
        "*********************************************\n");
      printf(
        "* set a connection mode to check CLOSED LOOP with DSVerifier (use: "
        "--connection-mode TYPE) *\n");
      printf(
        "**********************************************************************"
        "*****************************************\n");
      __DSVERIFIER_assert(0);
    }
  }
  if(12 == 0)
  {
    printf(
      "\n\n********************************************************************"
      "*******************\n");
    printf(
      "* set the property to check with DSVerifier (use: --property NAME) *\n");
    printf(
      "************************************************************************"
      "***************\n");
    __DSVERIFIER_assert(0);
  }
  if(
    (12 == 3) || (12 == 2) || (12 == 1) || (12 == 10) || (12 == 11) ||
    (12 == 4 || 12 == 5) || 12 == 6)
  {
    if(0 == 0)
    {
      printf(
        "\n\n******************************************************************"
        "**************************\n");
      printf(
        "* set a X_SIZE to use this property in DSVerifier (use: --x-size "
        "VALUE) *\n");
      printf(
        "**********************************************************************"
        "**********************\n");
      __DSVERIFIER_assert(0);
    }
    else
    {
      X_SIZE_VALUE = 0;
    }
  }
  if((0 == 0) && (12 != 9))
  {
    printf(
      "\n\n********************************************************************"
      "*************************\n");
    printf(
      "* set the realization to check with DSVerifier (use: --realization "
      "NAME) *\n");
    printf(
      "************************************************************************"
      "*********************\n");
    __DSVERIFIER_assert(0);
  }
  if(12 == 6 || 12 == 11)
  {
    if(impl.max_error == 0)
    {
      printf(
        "\n\n******************************************************************"
        "*****\n");
      printf("* provide the maximum expected error (use: impl.max_error) *\n");
      printf(
        "**********************************************************************"
        "*\n");
      __DSVERIFIER_assert(0);
    }
  }
  if(12 == 4 || 12 == 5)
  {
    if(12 == 5 || 12 == 4)
    {
      if(hw.clock == 0l)
      {
        printf("\n\n***************************\n");
        printf("* Clock could not be zero *\n");
        printf("***************************\n");
        __DSVERIFIER_assert(0);
      }
      hw.cycle = ((double)1.0 / hw.clock);
      if(hw.cycle < 0)
      {
        printf("\n\n*********************************************\n");
        printf("* The cycle time could not be representable *\n");
        printf("*********************************************\n");
        __DSVERIFIER_assert(0);
      }
      if(ds.sample_time == 0)
      {
        printf(
          "\n\n****************************************************************"
          "*************\n");
        printf(
          "* provide the sample time of the digital system (ds.sample_time) "
          "*\n");
        printf(
          "********************************************************************"
          "*********\n");
        __DSVERIFIER_assert(0);
      }
    }
  }
  if((0 == 7) || (0 == 8) || (0 == 9) || (0 == 10) || (0 == 11) || (0 == 12))
  {
    printf("\n\n******************************************\n");
    printf("* Temporarily the cascade modes are disabled *\n");
    printf("**********************************************\n");
    __DSVERIFIER_assert(0);
  }
}

void call_verification_task(void *verification_task)
{
  int i = 0;

  _Bool base_case_executed = 0;

  if(0 == 2)
  {
    for(i = 0; i < ds.b_size; i++)
    {
      if(ds.b_uncertainty[i] > 0)
      {
        double factor = ds.b_uncertainty[i];
        factor = factor < 0 ? factor * (-1) : factor;

        double min = ds.b[i] - factor;
        double max = ds.b[i] + factor;

        if((factor == 0) && (base_case_executed == 1))
        {
          continue;
        }
        else if((factor == 0) && (base_case_executed == 0))
        {
          base_case_executed = 1;
        }

        ds.b[i] = nondet_double();
        __DSVERIFIER_assume((ds.b[i] >= min) && (ds.b[i] <= max));
      }
    }

    for(i = 0; i < ds.a_size; i++)
    {
      if(ds.a_uncertainty[i] > 0)
      {
        double factor = ds.a_uncertainty[i];
        factor = factor < 0 ? factor * (-1) : factor;

        double min = ds.a[i] - factor;
        double max = ds.a[i] + factor;

        if((factor == 0) && (base_case_executed == 1))
        {
          continue;
        }
        else if((factor == 0) && (base_case_executed == 0))
        {
          base_case_executed = 1;
        }

        ds.a[i] = nondet_double();
        __DSVERIFIER_assume((ds.a[i] >= min) && (ds.a[i] <= max));
      }
    }
  }
  else
  {
    int i = 0;
    for(i = 0; i < ds.b_size; i++)
    {
      if(ds.b_uncertainty[i] > 0)
      {
        double factor = ((ds.b[i] * ds.b_uncertainty[i]) / 100);
        factor = factor < 0 ? factor * (-1) : factor;

        double min = ds.b[i] - factor;
        double max = ds.b[i] + factor;

        if((factor == 0) && (base_case_executed == 1))
        {
          continue;
        }
        else if((factor == 0) && (base_case_executed == 0))
        {
          base_case_executed = 1;
        }

        ds.b[i] = nondet_double();
        __DSVERIFIER_assume((ds.b[i] >= min) && (ds.b[i] <= max));
      }
    }

    for(i = 0; i < ds.a_size; i++)
    {
      if(ds.a_uncertainty[i] > 0)
      {
        double factor = ((ds.a[i] * ds.a_uncertainty[i]) / 100);
        factor = factor < 0 ? factor * (-1) : factor;

        double min = ds.a[i] - factor;
        double max = ds.a[i] + factor;

        if((factor == 0) && (base_case_executed == 1))
        {
          continue;
        }
        else if((factor == 0) && (base_case_executed == 0))
        {
          base_case_executed = 1;
        }

        ds.a[i] = nondet_double();
        __DSVERIFIER_assume((ds.a[i] >= min) && (ds.a[i] <= max));
      }
    }
  }

  ((void (*)())verification_task)();
}

void call_closedloop_verification_task(void *closedloop_verification_task)
{
  _Bool base_case_executed = 0;

  int i = 0;
  for(i = 0; i < plant.b_size; i++)
  {
    if(plant.b_uncertainty[i] > 0)
    {
      double factor = ((plant.b[i] * plant.b_uncertainty[i]) / 100);
      factor = factor < 0 ? factor * (-1) : factor;
      double min = plant.b[i] - factor;
      double max = plant.b[i] + factor;

      if((factor == 0) && (base_case_executed == 1))
      {
        continue;
      }
      else if((factor == 0) && (base_case_executed == 0))
      {
        base_case_executed = 1;
      }

      plant_cbmc.b[i] = nondet_double();
      __DSVERIFIER_assume((plant_cbmc.b[i] >= min) && (plant_cbmc.b[i] <= max));
    }
    else
    {
      plant_cbmc.b[i] = plant.b[i];
    }
  }

  for(i = 0; i < plant.a_size; i++)
  {
    if(plant.a_uncertainty[i] > 0)
    {
      double factor = ((plant.a[i] * plant.a_uncertainty[i]) / 100);
      factor = factor < 0 ? factor * (-1) : factor;

      double min = plant.a[i] - factor;
      double max = plant.a[i] + factor;

      if((factor == 0) && (base_case_executed == 1))
      {
        continue;
      }
      else if((factor == 0) && (base_case_executed == 0))
      {
        base_case_executed = 1;
      }

      plant_cbmc.a[i] = nondet_double();
      __DSVERIFIER_assume((plant_cbmc.a[i] >= min) && (plant_cbmc.a[i] <= max));
    }
    else
    {
      plant_cbmc.a[i] = plant.a[i];
    }
  }

  ((void (*)())closedloop_verification_task)();
}
# 2 "input.c" 2
digital_system_state_space _controller;
implementation impl = {.int_bits = 15, .frac_bits = 16};
int nStates = 3;
int nInputs = 1;
int nOutputs = 1;
double error_limit =
  1.0000000000000000000000000000000000000000000000000000000000000000;
double nondet_double(void);
void initials()
{
  _controller.A[0][0] =
    1.0008999999999999008792883614660240709781650000000000000000000000;
  _controller.A[0][1] =
    0.0010000000000000000208166817117216851329430000000000000000000000;
  _controller.A[0][2] =
    -0.0000168589999999999990136934774342947207510000000000000000000000;
  _controller.A[1][0] =
    1.8674999999999999378275106209912337362766270000000000000000000000;
  _controller.A[1][1] =
    1.0008999999999999008792883614660240709781650000000000000000000000;
  _controller.A[1][2] =
    -0.0335000000000000019984014443252817727625370000000000000000000000;
  _controller.A[2][0] =
    0.0000000000000000000000000000000000000000000000000000000000000000;
  _controller.A[2][1] =
    0.0000000000000000000000000000000000000000000000000000000000000000;
  _controller.A[2][2] =
    0.9585000000000000186517468137026298791170120000000000000000000000;
  _controller.B[0][0] =
    -0.0000000119979999999999993067194975493450220000000000000000000000;
  _controller.B[1][0] =
    0.0000358700000000000004626334038082546840090000000000000000000000;
  _controller.B[2][0] =
    0.0020999999999999998702426839969348293379880000000000000000000000;
  _controller.C[0][0] =
    1.0000000000000000000000000000000000000000000000000000000000000000;
  _controller.C[0][1] =
    0.0000000000000000000000000000000000000000000000000000000000000000;
  _controller.C[0][2] =
    0.0000000000000000000000000000000000000000000000000000000000000000;
  _controller.D[0][0] =
    0.0000000000000000000000000000000000000000000000000000000000000000;
  _controller.inputs[0][0] =
    1.0000000000000000000000000000000000000000000000000000000000000000;
#ifdef __NONDET_K
  _controller.K[0][0] = nondet_double();
  _controller.K[0][1] = nondet_double();
  _controller.K[0][2] = nondet_double();
#else
  _controller.K[0][0] =
    -1783500.0000000000000000000000000000000000000000000000000000000000000000;
  _controller.K[0][1] =
    -7670.0000000000000000000000000000000000000000000000000000000000000000;
  _controller.K[0][2] =
    438.7794000000000096406438387930393218994141000000000000000000000000;
#endif
}
