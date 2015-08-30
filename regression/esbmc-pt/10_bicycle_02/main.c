/* Provide Declarations */
#include <stdarg.h>
#include <setjmp.h>
/* get a declaration for alloca */
#if defined(__CYGWIN__) || defined(__MINGW32__)
#define  alloca(x) __builtin_alloca((x))
#define _alloca(x) __builtin_alloca((x))
#elif defined(__APPLE__)
extern void *__builtin_alloca(unsigned long);
#define alloca(x) __builtin_alloca(x)
#define longjmp _longjmp
#define setjmp _setjmp
#elif defined(__sun__)
#if defined(__sparcv9)
extern void *__builtin_alloca(unsigned long);
#else
extern void *__builtin_alloca(unsigned int);
#endif
#define alloca(x) __builtin_alloca(x)
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__) || defined(__arm__)
#define alloca(x) __builtin_alloca(x)
#elif defined(_MSC_VER)
#define inline _inline
#define alloca(x) _alloca(x)
#else
#include <alloca.h>
#endif

#ifndef __GNUC__  /* Can only support "linkonce" vars with GCC */
#define __attribute__(X)
#endif

#if defined(__GNUC__) && defined(__APPLE_CC__)
#define __EXTERNAL_WEAK__ __attribute__((weak_import))
#elif defined(__GNUC__)
#define __EXTERNAL_WEAK__ __attribute__((weak))
#else
#define __EXTERNAL_WEAK__
#endif

#if defined(__GNUC__) && defined(__APPLE_CC__)
#define __ATTRIBUTE_WEAK__
#elif defined(__GNUC__)
#define __ATTRIBUTE_WEAK__ __attribute__((weak))
#else
#define __ATTRIBUTE_WEAK__
#endif

#if defined(__GNUC__)
#define __HIDDEN__ __attribute__((visibility("hidden")))
#endif

#ifdef __GNUC__
#define LLVM_NAN(NanStr)   __builtin_nan(NanStr)   /* Double */
#define LLVM_NANF(NanStr)  __builtin_nanf(NanStr)  /* Float */
#define LLVM_NANS(NanStr)  __builtin_nans(NanStr)  /* Double */
#define LLVM_NANSF(NanStr) __builtin_nansf(NanStr) /* Float */
#define LLVM_INF           __builtin_inf()         /* Double */
#define LLVM_INFF          __builtin_inff()        /* Float */
#define LLVM_PREFETCH(addr,rw,locality) __builtin_prefetch(addr,rw,locality)
#define __ATTRIBUTE_CTOR__ __attribute__((constructor))
#define __ATTRIBUTE_DTOR__ __attribute__((destructor))
#define LLVM_ASM           __asm__
#else
#define LLVM_NAN(NanStr)   ((double)0.0)           /* Double */
#define LLVM_NANF(NanStr)  0.0F                    /* Float */
#define LLVM_NANS(NanStr)  ((double)0.0)           /* Double */
#define LLVM_NANSF(NanStr) 0.0F                    /* Float */
#define LLVM_INF           ((double)0.0)           /* Double */
#define LLVM_INFF          0.0F                    /* Float */
#define LLVM_PREFETCH(addr,rw,locality)            /* PREFETCH */
#define __ATTRIBUTE_CTOR__
#define __ATTRIBUTE_DTOR__
#define LLVM_ASM(X)
#endif

#if __GNUC__ < 4 /* Old GCC's, or compilers not GCC */ 
#define __builtin_stack_save() 0   /* not implemented */
#define __builtin_stack_restore(X) /* noop */
#endif

#if __GNUC__ && __LP64__ /* 128-bit integer types */
typedef int __attribute__((mode(TI))) llvmInt128;
typedef unsigned __attribute__((mode(TI))) llvmUInt128;
#endif

#define CODE_FOR_MAIN() /* Any target-specific code for main()*/

#ifndef __cplusplus
typedef unsigned char bool;
#endif


/* Support for floating point constants */
typedef unsigned long long ConstantDoubleTy;
typedef unsigned int        ConstantFloatTy;
typedef struct { unsigned long long f1; unsigned short f2; unsigned short pad[3]; } ConstantFP80Ty;
typedef struct { unsigned long long f1; unsigned long long f2; } ConstantFP128Ty;


/* Global Declarations */
/* Helper union for bitcasts */
typedef union {
  unsigned int Int32;
  unsigned long long Int64;
  float Float;
  double Double;
} llvmBitCastUnion;
/* Structure forward decls */
struct l_class_OC_Bicycle;
struct l_class_OC_Display;
struct l_class_OC_EmbeddedPC;
struct l_class_OC_Thread;
struct l_class_OC_std_KD__KD___basic_file;
struct l_class_OC_std_KD__KD_allocator;
struct l_class_OC_std_KD__KD_basic_filebuf;
struct l_class_OC_std_KD__KD_basic_ofstream;
struct l_class_OC_std_KD__KD_basic_ostream;
struct l_class_OC_std_KD__KD_basic_string;
struct l_class_OC_std_KD__KD_basic_stringbuf;
struct l_class_OC_std_KD__KD_basic_stringstream;
struct l_class_OC_std_KD__KD_codecvt;
struct l_struct_OC__MD_anonymous_AC_union_OD__KD__KD___pthread_mutex_s;
struct l_struct_OC__IO_FILE;
struct l_struct_OC__IO_marker;
struct l_struct_OC___locale_data;
struct l_struct_OC___locale_struct;
struct l_struct_OC_anon;
struct l_struct_OC_std_KD__KD_basic_string_MD_char_MC__AC_std_KD__KD_char_traits_MD_char_OD__MC__AC_std_KD__KD_allocator_MD_char_OD__AC__OD__KD__KD__Alloc_hider;
struct l_struct_OC_timespec;
struct l_union_OC_anon;
struct l_unnamed0;
struct l_unnamed1;
struct l_unnamed10;
struct l_unnamed11;
struct l_unnamed12;
struct l_unnamed13;
struct l_unnamed14;
struct l_unnamed15;
struct l_unnamed16;
struct l_unnamed17;
struct l_unnamed18;
struct l_unnamed19;
struct l_unnamed2;
struct l_unnamed20;
struct l_unnamed21;
struct l_unnamed22;
struct l_unnamed23;
struct l_unnamed24;
struct l_unnamed25;
struct l_unnamed26;
struct l_unnamed27;
struct l_unnamed28;
struct l_unnamed29;
struct l_unnamed3;
struct l_unnamed30;
struct l_unnamed31;
struct l_unnamed32;
struct l_unnamed33;
struct l_unnamed34;
struct l_unnamed4;
struct l_unnamed5;
struct l_unnamed6;
struct l_unnamed7;
struct l_unnamed8;
struct l_unnamed9;

/* Typedefs */
typedef struct l_class_OC_Bicycle l_class_OC_Bicycle;
typedef struct l_class_OC_Display l_class_OC_Display;
typedef struct l_class_OC_EmbeddedPC l_class_OC_EmbeddedPC;
typedef struct l_class_OC_Thread l_class_OC_Thread;
typedef struct l_class_OC_std_KD__KD___basic_file l_class_OC_std_KD__KD___basic_file;
typedef struct l_class_OC_std_KD__KD_allocator l_class_OC_std_KD__KD_allocator;
typedef struct l_class_OC_std_KD__KD_basic_filebuf l_class_OC_std_KD__KD_basic_filebuf;
typedef struct l_class_OC_std_KD__KD_basic_ofstream l_class_OC_std_KD__KD_basic_ofstream;
typedef struct l_class_OC_std_KD__KD_basic_ostream l_class_OC_std_KD__KD_basic_ostream;
typedef struct l_class_OC_std_KD__KD_basic_string l_class_OC_std_KD__KD_basic_string;
typedef struct l_class_OC_std_KD__KD_basic_stringbuf l_class_OC_std_KD__KD_basic_stringbuf;
typedef struct l_class_OC_std_KD__KD_basic_stringstream l_class_OC_std_KD__KD_basic_stringstream;
typedef struct l_class_OC_std_KD__KD_codecvt l_class_OC_std_KD__KD_codecvt;
typedef struct l_struct_OC__MD_anonymous_AC_union_OD__KD__KD___pthread_mutex_s l_struct_OC__MD_anonymous_AC_union_OD__KD__KD___pthread_mutex_s;
typedef struct l_struct_OC__IO_FILE l_struct_OC__IO_FILE;
typedef struct l_struct_OC__IO_marker l_struct_OC__IO_marker;
typedef struct l_struct_OC___locale_data l_struct_OC___locale_data;
typedef struct l_struct_OC___locale_struct l_struct_OC___locale_struct;
typedef struct l_struct_OC_anon l_struct_OC_anon;
typedef struct l_struct_OC_std_KD__KD_basic_string_MD_char_MC__AC_std_KD__KD_char_traits_MD_char_OD__MC__AC_std_KD__KD_allocator_MD_char_OD__AC__OD__KD__KD__Alloc_hider l_struct_OC_std_KD__KD_basic_string_MD_char_MC__AC_std_KD__KD_char_traits_MD_char_OD__MC__AC_std_KD__KD_allocator_MD_char_OD__AC__OD__KD__KD__Alloc_hider;
typedef struct l_struct_OC_timespec l_struct_OC_timespec;
typedef struct l_union_OC_anon l_union_OC_anon;
typedef struct l_unnamed0 l_unnamed0;
typedef struct l_unnamed1 l_unnamed1;
typedef struct l_unnamed10 l_unnamed10;
typedef struct l_unnamed11 l_unnamed11;
typedef struct l_unnamed12 l_unnamed12;
typedef struct l_unnamed13 l_unnamed13;
typedef struct l_unnamed14 l_unnamed14;
typedef struct l_unnamed15 l_unnamed15;
typedef struct l_unnamed16 l_unnamed16;
typedef struct l_unnamed17 l_unnamed17;
typedef struct l_unnamed18 l_unnamed18;
typedef struct l_unnamed19 l_unnamed19;
typedef struct l_unnamed2 l_unnamed2;
typedef struct l_unnamed20 l_unnamed20;
typedef struct l_unnamed21 l_unnamed21;
typedef struct l_unnamed22 l_unnamed22;
typedef struct l_unnamed23 l_unnamed23;
typedef struct l_unnamed24 l_unnamed24;
typedef struct l_unnamed25 l_unnamed25;
typedef struct l_unnamed26 l_unnamed26;
typedef struct l_unnamed27 l_unnamed27;
typedef struct l_unnamed28 l_unnamed28;
typedef struct l_unnamed29 l_unnamed29;
typedef struct l_unnamed3 l_unnamed3;
typedef struct l_unnamed30 l_unnamed30;
typedef struct l_unnamed31 l_unnamed31;
typedef struct l_unnamed32 l_unnamed32;
typedef struct l_unnamed33 l_unnamed33;
typedef struct l_unnamed34 l_unnamed34;
typedef struct l_unnamed4 l_unnamed4;
typedef struct l_unnamed5 l_unnamed5;
typedef struct l_unnamed6 l_unnamed6;
typedef struct l_unnamed7 l_unnamed7;
typedef struct l_unnamed8 l_unnamed8;
typedef struct l_unnamed9 l_unnamed9;

/* Structure contents */
struct l_unnamed19 { unsigned char array[144]; };

struct l_class_OC_Bicycle {
  struct l_unnamed19 field0;
  struct l_class_OC_EmbeddedPC *field1;
};

struct l_unnamed20 { unsigned char array[4]; };

struct l_unnamed10 { unsigned char array[32]; };

struct l_union_OC_anon {
  unsigned int field0;
};

struct l_struct_OC__MD_anonymous_AC_union_OD__KD__KD___pthread_mutex_s {
  unsigned int field0;
  unsigned int field1;
  unsigned int field2;
  unsigned int field3;
  unsigned int field4;
  struct l_union_OC_anon field5;
};

struct l_unnamed32 {
  struct l_struct_OC__MD_anonymous_AC_union_OD__KD__KD___pthread_mutex_s field0;
};

struct l_class_OC_std_KD__KD___basic_file {
  struct l_struct_OC__IO_FILE *field0;
  unsigned char field1;
};

struct l_unnamed8 {
  unsigned int field0;
  struct l_union_OC_anon field1;
};

struct l_class_OC_std_KD__KD_basic_filebuf {
  struct l_unnamed10 field0;
  struct l_unnamed32 field1;
  struct l_class_OC_std_KD__KD___basic_file field2;
  unsigned int field3;
  struct l_unnamed8 field4;
  struct l_unnamed8 field5;
  struct l_unnamed8 field6;
  unsigned char *field7;
  unsigned int field8;
  unsigned char field9;
  unsigned char field10;
  unsigned char field11;
  unsigned char field12;
  unsigned char *field13;
  unsigned char *field14;
  unsigned char field15;
  struct l_class_OC_std_KD__KD_codecvt *field16;
  unsigned char *field17;
  unsigned int field18;
  unsigned char *field19;
  unsigned char *field20;
};

struct l_unnamed15 { unsigned char array[136]; };

struct l_class_OC_std_KD__KD_basic_ofstream {
  struct l_unnamed20 field0;
  struct l_class_OC_std_KD__KD_basic_filebuf field1;
  struct l_unnamed15 field2;
};

struct l_struct_OC_std_KD__KD_basic_string_MD_char_MC__AC_std_KD__KD_char_traits_MD_char_OD__MC__AC_std_KD__KD_allocator_MD_char_OD__AC__OD__KD__KD__Alloc_hider {
  unsigned char *field0;
};

struct l_class_OC_std_KD__KD_basic_string {
  struct l_struct_OC_std_KD__KD_basic_string_MD_char_MC__AC_std_KD__KD_char_traits_MD_char_OD__MC__AC_std_KD__KD_allocator_MD_char_OD__AC__OD__KD__KD__Alloc_hider field0;
};

struct l_class_OC_Display {
  struct l_unnamed19 field0;
  struct l_class_OC_std_KD__KD_basic_ofstream field1;
  struct l_class_OC_std_KD__KD_basic_string field2;
  unsigned int field3;
};

struct l_struct_OC_timespec {
  unsigned int field0;
  unsigned int field1;
};

struct l_class_OC_EmbeddedPC {
  struct l_unnamed32 field0;
  unsigned int field1;
  unsigned char field2;
  struct l_struct_OC_timespec field3;
  struct l_struct_OC_timespec field4;
  float field5;
  float field6;
  float field7;
  struct l_class_OC_Display *field8;
};

struct l_struct_OC_anon {
  unsigned int field0;
  unsigned int field1;
  unsigned long long field2;
  unsigned long long field3;
  unsigned long long field4;
  unsigned char *field5;
  unsigned int field6;
  unsigned int field7;
};

struct l_unnamed26 {
  struct l_struct_OC_anon field0;
  struct l_unnamed20 field1;
};

struct l_unnamed27 {
  unsigned int field0;
  struct l_unnamed10 field1;
};

struct l_class_OC_Thread {
  unsigned int  (**field0) ( int, ...);
  struct l_unnamed32 field1;
  struct l_unnamed32 field2;
  struct l_unnamed26 field3;
  unsigned char field4;
  struct l_unnamed27 field5;
  unsigned int field6;
};

struct l_class_OC_std_KD__KD_allocator {
  unsigned char field0;
};

struct l_class_OC_std_KD__KD_basic_ostream {
  unsigned int  (**field0) ( int, ...);
  struct l_unnamed15 field1;
};

struct l_class_OC_std_KD__KD_basic_stringbuf {
  struct l_unnamed10 field0;
  unsigned int field1;
  struct l_class_OC_std_KD__KD_basic_string field2;
};

struct l_unnamed13 { unsigned char array[12]; };

struct l_class_OC_std_KD__KD_basic_stringstream {
  struct l_unnamed13 field0;
  struct l_class_OC_std_KD__KD_basic_stringbuf field1;
  struct l_unnamed15 field2;
};

struct l_unnamed31 { unsigned char array[8]; };

struct l_class_OC_std_KD__KD_codecvt {
  struct l_unnamed31 field0;
  struct l_struct_OC___locale_struct *field1;
};

struct l_unnamed30 { unsigned char array[1]; };

struct l_unnamed24 { unsigned char array[40]; };

struct l_struct_OC__IO_FILE {
  unsigned int field0;
  unsigned char *field1;
  unsigned char *field2;
  unsigned char *field3;
  unsigned char *field4;
  unsigned char *field5;
  unsigned char *field6;
  unsigned char *field7;
  unsigned char *field8;
  unsigned char *field9;
  unsigned char *field10;
  unsigned char *field11;
  struct l_struct_OC__IO_marker *field12;
  struct l_struct_OC__IO_FILE *field13;
  unsigned int field14;
  unsigned int field15;
  unsigned int field16;
  unsigned short field17;
  unsigned char field18;
  struct l_unnamed30 field19;
  unsigned char *field20;
  unsigned long long field21;
  unsigned char *field22;
  unsigned char *field23;
  unsigned char *field24;
  unsigned char *field25;
  unsigned int field26;
  unsigned int field27;
  struct l_unnamed24 field28;
};

struct l_struct_OC__IO_marker {
  struct l_struct_OC__IO_marker *field0;
  struct l_struct_OC__IO_FILE *field1;
  unsigned int field2;
};

struct l_unnamed22 { struct l_struct_OC___locale_data *array[13]; };

struct l_unnamed34 { unsigned char *array[13]; };

struct l_struct_OC___locale_struct {
  struct l_unnamed22 field0;
  unsigned short *field1;
  unsigned int *field2;
  unsigned int *field3;
  struct l_unnamed34 field4;
};

struct l_unnamed0 { unsigned char array[24]; };

struct l_unnamed1 { unsigned char array[29]; };

struct l_unnamed11 { unsigned char array[22]; };

struct l_unnamed12 { unsigned char array[23]; };

struct l_unnamed14 { unsigned char array[26]; };

struct l_unnamed28 {
  unsigned int field0;
  void  (*field1) (void);
};

struct l_unnamed16 { struct l_unnamed28 array[3]; };

struct l_unnamed17 { unsigned char array[11]; };

struct l_unnamed18 {
  unsigned char *field0;
  unsigned char *field1;
};

struct l_unnamed2 { unsigned char array[5]; };

struct l_unnamed21 { unsigned char array[3]; };

struct l_unnamed23 { unsigned char array[2]; };

struct l_unnamed25 {
  unsigned char *field0;
  unsigned char *field1;
  unsigned char *field2;
};

struct l_unnamed29 { unsigned char array[9]; };

struct l_unnamed3 { unsigned char array[16]; };

struct l_unnamed33 { unsigned char *array[6]; };

struct l_unnamed4 { unsigned char array[7]; };

struct l_unnamed5 { unsigned char array[53]; };

struct l_unnamed6 { unsigned char array[20]; };

struct l_unnamed7 { unsigned char array[19]; };

struct l_unnamed9 { unsigned char array[21]; };


/* External Global Variable Declarations */
extern unsigned char *__dso_handle;
extern unsigned char *_ZTVN10__cxxabiv120__si_class_type_infoE;
extern struct l_class_OC_std_KD__KD_basic_ostream _ZSt4cout;
extern unsigned char *_ZTVN10__cxxabiv117__class_type_infoE;

/* Function Declarations */
double fmod(double, double);
float fmodf(float, float);
long double fmodl(long double, long double);
void _ZNSt8ios_base4InitC1Ev(struct l_class_OC_std_KD__KD_allocator *);
void _ZNSt8ios_base4InitD1Ev(struct l_class_OC_std_KD__KD_allocator *);
unsigned int __cxa_atexit(void  (*) (unsigned char *), unsigned char *, unsigned char *);
static void _ZN7Bicycle3runEv(struct l_class_OC_Bicycle *this);
unsigned int rand(void);
static void _ZN7BicycleD1Ev(struct l_class_OC_Bicycle *this);
static void _ZN7BicycleD0Ev(struct l_class_OC_Bicycle *this);
unsigned int __gxx_personality_v0(int vararg_dummy_arg,...);
void _Unwind_Resume_or_Rethrow(unsigned char *);
void _ZdlPv(unsigned char *);
static void _GLOBAL__I_a(void) __ATTRIBUTE_CTOR__;
void _ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1Ev(struct l_class_OC_std_KD__KD_basic_ofstream *);
void _ZNSsC1ERKSs(struct l_class_OC_std_KD__KD_basic_string *, struct l_class_OC_std_KD__KD_basic_string *);
void _ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev(struct l_class_OC_std_KD__KD_basic_ofstream *);
void _ZSt9terminatev(void);
struct l_class_OC_std_KD__KD_basic_string *_ZNSsaSERKSs(struct l_class_OC_std_KD__KD_basic_string *, struct l_class_OC_std_KD__KD_basic_string *);
static void _ZN7Display3runEv(struct l_class_OC_Display *this);
void _ZNSt14basic_ofstreamIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode(struct l_class_OC_std_KD__KD_basic_ofstream *, unsigned char *, unsigned int );
struct l_class_OC_std_KD__KD_basic_ostream *_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(struct l_class_OC_std_KD__KD_basic_ostream *, unsigned char *);
struct l_class_OC_std_KD__KD_basic_ostream *_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(struct l_class_OC_std_KD__KD_basic_ostream *, struct l_class_OC_std_KD__KD_basic_string *);
void _ZNSt14basic_ofstreamIcSt11char_traitsIcEE5closeEv(struct l_class_OC_std_KD__KD_basic_ofstream *);
static void _ZN7DisplayD1Ev(struct l_class_OC_Display *this);
static void _ZN7DisplayD0Ev(struct l_class_OC_Display *this);
void _ZNSsD1Ev(struct l_class_OC_std_KD__KD_basic_string *);
unsigned int gettimeofday(struct l_struct_OC_timespec *, struct l_struct_OC_timespec *);
unsigned int pthread_mutex_init(struct l_unnamed32 *, struct l_union_OC_anon *);
void _ZNSt18basic_stringstreamIcSt11char_traitsIcESaIcEEC1ESt13_Ios_Openmode(struct l_class_OC_std_KD__KD_basic_stringstream *, unsigned int );
void _ZNSsC1EPKcRKSaIcE(struct l_class_OC_std_KD__KD_basic_string *, unsigned char *, struct l_class_OC_std_KD__KD_allocator *);
void _ZNSaIcEC1Ev(struct l_class_OC_std_KD__KD_allocator *);
void _ZNSaIcED1Ev(struct l_class_OC_std_KD__KD_allocator *);
struct l_class_OC_std_KD__KD_basic_ostream *_ZNSolsEf(struct l_class_OC_std_KD__KD_basic_ostream *, float );
unsigned char *_Znwj(unsigned int );
struct l_class_OC_std_KD__KD_basic_string _ZNKSt18basic_stringstreamIcSt11char_traitsIcESaIcEE3strEv(struct l_class_OC_std_KD__KD_basic_stringstream *);
void _ZNSt18basic_stringstreamIcSt11char_traitsIcESaIcEED1Ev(struct l_class_OC_std_KD__KD_basic_stringstream *);
static void _GLOBAL__I_a7(void) __ATTRIBUTE_CTOR__;
unsigned int main(void);
struct l_class_OC_std_KD__KD_basic_ostream *_ZNSolsEPFRSoS_E(struct l_class_OC_std_KD__KD_basic_ostream *, struct l_class_OC_std_KD__KD_basic_ostream * (*) (struct l_class_OC_std_KD__KD_basic_ostream *));
struct l_class_OC_std_KD__KD_basic_ostream *_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(struct l_class_OC_std_KD__KD_basic_ostream *);
void __assert_fail(unsigned char *, unsigned char *, unsigned int , unsigned char *);
static void _GLOBAL__I_a25(void) __ATTRIBUTE_CTOR__;
unsigned int pthread_cond_init(struct l_unnamed26 *, struct l_union_OC_anon *);
unsigned int pthread_attr_init(struct l_unnamed27 *);
unsigned int pthread_attr_setdetachstate(struct l_unnamed27 *, unsigned int );
unsigned int pthread_attr_setscope(struct l_unnamed27 *, unsigned int );
static void _ZN6ThreadD0Ev(struct l_class_OC_Thread *this);
static void _ZN6ThreadD2Ev(struct l_class_OC_Thread *this);
unsigned int pthread_cond_signal(struct l_unnamed26 *);
unsigned int pthread_cond_destroy(struct l_unnamed26 *);
unsigned int pthread_mutex_unlock(struct l_unnamed32 *);
unsigned int pthread_mutex_destroy(struct l_unnamed32 *);
unsigned int pthread_attr_destroy(struct l_unnamed27 *);
static unsigned char *_ZN6Thread8functionEPv(unsigned char *ptr);
void pthread_exit(unsigned char *);
static void _ZN6Thread5startEv(struct l_class_OC_Thread *this);
unsigned int pthread_create(unsigned int *, struct l_unnamed27 *, unsigned char * (*) (unsigned char *), unsigned char *);
unsigned int pthread_detach(unsigned int );
unsigned int pthread_mutex_lock(struct l_unnamed32 *);
unsigned int pthread_cond_timedwait(struct l_unnamed26 *, struct l_unnamed32 *, struct l_struct_OC_timespec *);
void __cxa_pure_virtual(void);
void abort(void);


/* Global Variable Declarations */
static struct l_class_OC_std_KD__KD_allocator _ZStL8__ioinit;
static struct l_unnamed33 _ZTV7Bicycle;
static struct l_unnamed29 _ZTS7Bicycle;
static struct l_unnamed25 _ZTI7Bicycle;
static struct l_unnamed33 _ZTV7Display;
static struct l_unnamed31 _OC_str;
static struct l_unnamed21 _OC_str1;
static struct l_unnamed29 _ZTS7Display;
static struct l_unnamed25 _ZTI7Display;
static struct l_class_OC_std_KD__KD_allocator _ZStL8__ioinit1;
static struct l_unnamed6 _OC_str2;
static struct l_unnamed23 _OC_str13;
static struct l_unnamed31 _OC_str24;
static struct l_unnamed3 _OC_str8;
static struct l_unnamed2 _OC_str9;
static struct l_unnamed0 _OC_str10;
static struct l_unnamed1 _OC_str11;
static struct l_unnamed21 _OC_str12;
static struct l_class_OC_std_KD__KD_allocator _ZStL8__ioinit8;
static struct l_unnamed13 _OC_str14;
static struct l_unnamed4 _OC_str115;
static struct l_unnamed5 _OC_str216;
static struct l_unnamed7 _OC_str317;
static struct l_unnamed9 _OC_str418;
static struct l_unnamed11 _OC_str519;
static struct l_unnamed12 _OC_str620;
static struct l_unnamed14 _OC_str721;
static struct l_unnamed23 _OC_str822;
static struct l_unnamed29 _OC_str923;
static struct l_unnamed17 __PRETTY_FUNCTION___OC_main;
static struct l_unnamed33 _ZTV6Thread;
static struct l_unnamed31 _ZTS6Thread;
static struct l_unnamed18 _ZTI6Thread;


/* Global Variable Definitions and Initialization */
static struct l_class_OC_std_KD__KD_allocator _ZStL8__ioinit;
static struct l_unnamed33 _ZTV7Bicycle = { { ((unsigned char *)/*NULL*/0), ((unsigned char *)(&_ZTI7Bicycle)), ((unsigned char *)_ZN7Bicycle3runEv), ((unsigned char *)_ZN7BicycleD1Ev), ((unsigned char *)_ZN7BicycleD0Ev), ((unsigned char *)_ZN6Thread5startEv) } };
static struct l_unnamed29 _ZTS7Bicycle = { "7Bicycle" };
static struct l_unnamed25 _ZTI7Bicycle = { ((unsigned char *)((&(&_ZTVN10__cxxabiv120__si_class_type_infoE)[((signed int )2u)]))), ((&_ZTS7Bicycle.array[((signed int )0u)])), ((unsigned char *)(&_ZTI6Thread)) };
static struct l_unnamed33 _ZTV7Display = { { ((unsigned char *)/*NULL*/0), ((unsigned char *)(&_ZTI7Display)), ((unsigned char *)_ZN7Display3runEv), ((unsigned char *)_ZN7DisplayD1Ev), ((unsigned char *)_ZN7DisplayD0Ev), ((unsigned char *)_ZN6Thread5startEv) } };
static struct l_unnamed31 _OC_str = { "Display" };
static struct l_unnamed21 _OC_str1 = { ".\n" };
static struct l_unnamed29 _ZTS7Display = { "7Display" };
static struct l_unnamed25 _ZTI7Display = { ((unsigned char *)((&(&_ZTVN10__cxxabiv120__si_class_type_infoE)[((signed int )2u)]))), ((&_ZTS7Display.array[((signed int )0u)])), ((unsigned char *)(&_ZTI6Thread)) };
static struct l_class_OC_std_KD__KD_allocator _ZStL8__ioinit1;
static struct l_unnamed6 _OC_str2 = { "Distance traveled: " };
static struct l_unnamed23 _OC_str13 = { " " };
static struct l_unnamed31 _OC_str24 = { " meters" };
static struct l_unnamed3 _OC_str8 = { "Current speed: " };
static struct l_unnamed2 _OC_str9 = { " m/s" };
static struct l_unnamed0 _OC_str10 = { "Total distance so far: " };
static struct l_unnamed1 _OC_str11 = { "Time elapsed since started: " };
static struct l_unnamed21 _OC_str12 = { " s" };
static struct l_class_OC_std_KD__KD_allocator _ZStL8__ioinit8;
static struct l_unnamed13 _OC_str14 = { "Booting ..." };
static struct l_unnamed4 _OC_str115 = { " Done!" };
static struct l_unnamed5 _OC_str216 = { "----------------------------------------------------" };
static struct l_unnamed7 _OC_str317 = { "You are now biking" };
static struct l_unnamed9 _OC_str418 = { "Choose your option: " };
static struct l_unnamed11 _OC_str519 = { "1 - Press Mode Button" };
static struct l_unnamed12 _OC_str620 = { "2 - Press Reset Button" };
static struct l_unnamed14 _OC_str721 = { "3 - Remove PC's Batteries" };
static struct l_unnamed23 _OC_str822 = { "0" };
static struct l_unnamed29 _OC_str923 = { "main.cpp" };
static struct l_unnamed17 __PRETTY_FUNCTION___OC_main = { "int main()" };
static struct l_unnamed33 _ZTV6Thread = { { ((unsigned char *)/*NULL*/0), ((unsigned char *)(&_ZTI6Thread)), ((unsigned char *)__cxa_pure_virtual), ((unsigned char *)_ZN6ThreadD2Ev), ((unsigned char *)_ZN6ThreadD0Ev), ((unsigned char *)_ZN6Thread5startEv) } };
static struct l_unnamed31 _ZTS6Thread = { "6Thread" };
static struct l_unnamed18 _ZTI6Thread = { ((unsigned char *)((&(&_ZTVN10__cxxabiv117__class_type_infoE)[((signed int )2u)]))), ((&_ZTS6Thread.array[((signed int )0u)])) };


/* Function Bodies */
static inline int llvm_fcmp_ord(double X, double Y) { return X == X && Y == Y; }
static inline int llvm_fcmp_uno(double X, double Y) { return X != X || Y != Y; }
static inline int llvm_fcmp_ueq(double X, double Y) { return X == Y || llvm_fcmp_uno(X, Y); }
static inline int llvm_fcmp_une(double X, double Y) { return X != Y; }
static inline int llvm_fcmp_ult(double X, double Y) { return X <  Y || llvm_fcmp_uno(X, Y); }
static inline int llvm_fcmp_ugt(double X, double Y) { return X >  Y || llvm_fcmp_uno(X, Y); }
static inline int llvm_fcmp_ule(double X, double Y) { return X <= Y || llvm_fcmp_uno(X, Y); }
static inline int llvm_fcmp_uge(double X, double Y) { return X >= Y || llvm_fcmp_uno(X, Y); }
static inline int llvm_fcmp_oeq(double X, double Y) { return X == Y ; }
static inline int llvm_fcmp_one(double X, double Y) { return X != Y && llvm_fcmp_ord(X, Y); }
static inline int llvm_fcmp_olt(double X, double Y) { return X <  Y ; }
static inline int llvm_fcmp_ogt(double X, double Y) { return X >  Y ; }
static inline int llvm_fcmp_ole(double X, double Y) { return X <= Y ; }
static inline int llvm_fcmp_oge(double X, double Y) { return X >= Y ; }
static const ConstantFloatTy FPConstant0 = 0x7F800000U;    /* inf */

static void _ZN7Bicycle3runEv(struct l_class_OC_Bicycle *this) {
  struct l_struct_OC_timespec time_2e_i;    /* Address-exposed local */
  struct l_struct_OC_timespec timeOut_2e_i;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_stringstream msg_2e_i_2e_i;    /* Address-exposed local */
  struct l_struct_OC_timespec now_2e_i_2e_i;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string tmp__1;    /* Address-exposed local */
  struct l_struct_OC_timespec now_2e_i;    /* Address-exposed local */
  struct l_class_OC_EmbeddedPC **tmp__2;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__3;
  unsigned int *tmp__4;
  unsigned int *tmp__5;
  unsigned int *tmp__6;
  unsigned int *tmp__7;
  unsigned int *tmp__8;
  struct l_unnamed32 *tmp__9;
  struct l_unnamed26 *tmp__10;
  unsigned int tmp__11;
  unsigned int tmp__12;
  unsigned int tmp__13;
  unsigned int tmp__14;
  unsigned int tmp__15;
  unsigned int micro_2e_0_2e_i;
  unsigned int micro_2e_0_2e_i__PHI_TEMPORARY;
  unsigned int tmp__16;
  unsigned int tmp__17;
  unsigned int tmp__18;
  unsigned int tmp__19;
  struct l_class_OC_EmbeddedPC *tmp__20;
  unsigned int tmp__21;
  float *tmp__22;
  float tmp__23;
  float *tmp__24;
  float tmp__25;
  float *tmp__26;
  unsigned int tmp__27;
  struct l_class_OC_Display **tmp__28;
  struct l_class_OC_Display *tmp__29;
  unsigned int tmp__30;
  unsigned int tmp__31;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__32;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__33;
  float tmp__34;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__35;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__36;
  struct l_class_OC_Display *tmp__37;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__38;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__39;
  float tmp__40;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__41;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__42;
  struct l_class_OC_Display *tmp__43;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__44;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__45;
  float tmp__46;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__47;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__48;
  struct l_class_OC_Display *tmp__49;
  unsigned int tmp__50;
  unsigned int tmp__51;
  unsigned int tmp__52;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__53;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__54;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__55;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__56;
  struct l_class_OC_Display *tmp__57;
  struct l_class_OC_Display *tmp__58;
  struct l_class_OC_std_KD__KD_basic_string *tmp__59;
  struct l_class_OC_Display *tmp__60;
  unsigned int tmp__61;

  tmp__2 = (&this->field1);
  tmp__3 = ((struct l_class_OC_std_KD__KD_basic_ostream *)((&msg_2e_i_2e_i.field0.array[((signed int )8u)])));
  tmp__4 = (&now_2e_i_2e_i.field0);
  tmp__5 = (&time_2e_i.field0);
  tmp__6 = (&timeOut_2e_i.field0);
  tmp__7 = (&time_2e_i.field1);
  tmp__8 = (&timeOut_2e_i.field1);
  tmp__9 = ((struct l_unnamed32 *)((&this->field0.array[((signed int )28u)])));
  tmp__10 = ((struct l_unnamed26 *)((&this->field0.array[((signed int )52u)])));
  goto _2e_backedge;

  do {     /* Syntactic loop '.backedge' to make GCC happy */
_2e_backedge:
  tmp__11 = gettimeofday((&time_2e_i), ((struct l_struct_OC_timespec *)/*NULL*/0));
  tmp__12 = *tmp__5;
  *tmp__6 = tmp__12;
  tmp__13 = *tmp__7;
  tmp__14 = ((unsigned int )(((unsigned int )tmp__13) + ((unsigned int )100000u)));
  if ((((signed int )tmp__14) > ((signed int )1000000u))) {
    goto tmp__62;
  } else {
    micro_2e_0_2e_i__PHI_TEMPORARY = tmp__14;   /* for PHI node */
    goto _ZN6Thread6msleepEl_2e_exit;
  }

_ZN6Thread6msleepEl_2e_exit:
  micro_2e_0_2e_i = micro_2e_0_2e_i__PHI_TEMPORARY;
  *tmp__8 = (((unsigned int )(((unsigned int )micro_2e_0_2e_i) * ((unsigned int )1000u))));
  tmp__16 = pthread_mutex_lock(tmp__9);
  tmp__17 = pthread_cond_timedwait(tmp__10, tmp__9, (&timeOut_2e_i));
  tmp__18 = pthread_mutex_unlock(tmp__9);
  tmp__19 = rand();
  if ((((signed int )(((unsigned int )(((unsigned int )(((signed int )(((signed int )tmp__19) % ((signed int )100u))))) + ((unsigned int )1u))))) < ((signed int )31u))) {
    goto tmp__63;
  } else {
    goto _2e_backedge;
  }

tmp__62:
  *tmp__6 = (((unsigned int )(((unsigned int )tmp__12) + ((unsigned int )1u))));
  tmp__15 = ((unsigned int )(((unsigned int )tmp__13) + ((unsigned int )4294067296u)));
  micro_2e_0_2e_i__PHI_TEMPORARY = tmp__15;   /* for PHI node */
  goto _ZN6Thread6msleepEl_2e_exit;

_ZN7Display14setRefreshRateEi_2e_exit_2e_i_2e_i:
  tmp__58 = *tmp__28;
  tmp__1 = _ZNKSt18basic_stringstreamIcSt11char_traitsIcESaIcEE3strEv((&msg_2e_i_2e_i));
  tmp__59 = _ZNSsaSERKSs(((&tmp__58->field2)), (&tmp__1));
  _ZNSsD1Ev((&tmp__1));
  tmp__60 = *tmp__28;
  tmp__61 = pthread_mutex_unlock((((struct l_unnamed32 *)((&tmp__60->field0.array[((signed int )4u)])))));
  _ZNSt18basic_stringstreamIcSt11char_traitsIcESaIcEED1Ev((&msg_2e_i_2e_i));
  goto _2e_backedge;

tmp__63:
  tmp__20 = *tmp__2;
  tmp__21 = gettimeofday((&now_2e_i), ((struct l_struct_OC_timespec *)/*NULL*/0));
  tmp__22 = (&tmp__20->field5);
  tmp__23 = *tmp__22;
  *tmp__22 = (((float )(tmp__23 + 1.000000)));
  tmp__24 = (&tmp__20->field6);
  tmp__25 = *tmp__24;
  *tmp__24 = (((float )(tmp__25 + 1.000000)));
  tmp__26 = (&tmp__20->field7);
  *tmp__26 = (*(float*)&FPConstant0);
  tmp__27 = gettimeofday(((&tmp__20->field3)), ((struct l_struct_OC_timespec *)/*NULL*/0));
  tmp__28 = (&tmp__20->field8);
  tmp__29 = *tmp__28;
  tmp__30 = pthread_mutex_lock((((struct l_unnamed32 *)((&tmp__29->field0.array[((signed int )4u)])))));
  _ZNSt18basic_stringstreamIcSt11char_traitsIcESaIcEEC1ESt13_Ios_Openmode((&msg_2e_i_2e_i), 24u);
  tmp__31 = *((&tmp__20->field1));
  switch (tmp__31) {
  default:
    goto _ZN7Display14setRefreshRateEi_2e_exit_2e_i_2e_i;
;
  case 0u:
    goto tmp__64;
    break;
  case 1u:
    goto tmp__65;
  case 2u:
    goto _ZN7Display14setRefreshRateEi_2e_exit2_2e_i_2e_i;
  case 3u:
    goto _ZN7Display14setRefreshRateEi_2e_exit1_2e_i_2e_i;
  }
tmp__64:
  tmp__32 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__3, ((&_OC_str2.array[((signed int )0u)])));
  tmp__33 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__3, ((&_OC_str13.array[((signed int )0u)])));
  tmp__34 = *tmp__22;
  tmp__35 = _ZNSolsEf(tmp__33, tmp__34);
  tmp__36 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__35, ((&_OC_str24.array[((signed int )0u)])));
  tmp__37 = *tmp__28;
  *((&tmp__37->field3)) = 200u;
  goto _ZN7Display14setRefreshRateEi_2e_exit_2e_i_2e_i;

tmp__65:
  tmp__38 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__3, ((&_OC_str8.array[((signed int )0u)])));
  tmp__39 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__3, ((&_OC_str13.array[((signed int )0u)])));
  tmp__40 = *tmp__26;
  tmp__41 = _ZNSolsEf(tmp__39, tmp__40);
  tmp__42 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__41, ((&_OC_str9.array[((signed int )0u)])));
  tmp__43 = *tmp__28;
  *((&tmp__43->field3)) = 100u;
  goto _ZN7Display14setRefreshRateEi_2e_exit_2e_i_2e_i;

_ZN7Display14setRefreshRateEi_2e_exit2_2e_i_2e_i:
  tmp__44 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__3, ((&_OC_str10.array[((signed int )0u)])));
  tmp__45 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__3, ((&_OC_str13.array[((signed int )0u)])));
  tmp__46 = *tmp__24;
  tmp__47 = _ZNSolsEf(tmp__45, tmp__46);
  tmp__48 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__47, ((&_OC_str24.array[((signed int )0u)])));
  tmp__49 = *tmp__28;
  *((&tmp__49->field3)) = 500u;
  goto _ZN7Display14setRefreshRateEi_2e_exit_2e_i_2e_i;

_ZN7Display14setRefreshRateEi_2e_exit1_2e_i_2e_i:
  tmp__50 = gettimeofday((&now_2e_i_2e_i), ((struct l_struct_OC_timespec *)/*NULL*/0));
  tmp__51 = *tmp__4;
  tmp__52 = *((&tmp__20->field4.field0));
  tmp__53 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__3, ((&_OC_str11.array[((signed int )0u)])));
  tmp__54 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__3, ((&_OC_str13.array[((signed int )0u)])));
  tmp__55 = _ZNSolsEf(tmp__54, (((float )(signed int )(((unsigned int )(((unsigned int )tmp__51) - ((unsigned int )tmp__52)))))));
  tmp__56 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__55, ((&_OC_str12.array[((signed int )0u)])));
  tmp__57 = *tmp__28;
  *((&tmp__57->field3)) = 500u;
  goto _ZN7Display14setRefreshRateEi_2e_exit_2e_i_2e_i;

  } while (1); /* end of syntactic loop '.backedge' */
}


static void _ZN7BicycleD1Ev(struct l_class_OC_Bicycle *this) {
  struct l_unnamed26 *tmp__66;
  unsigned int tmp__67;
  unsigned int tmp__68;
  struct l_unnamed32 *tmp__69;
  unsigned int tmp__70;
  unsigned int tmp__71;
  struct l_unnamed32 *tmp__72;
  unsigned int tmp__73;
  unsigned int tmp__74;
  unsigned int tmp__75;

  *(((unsigned int  (***) ( int, ...))this)) = ((unsigned int  (**) ( int, ...))((&_ZTV6Thread.array[((signed int )2u)])));
  tmp__66 = ((struct l_unnamed26 *)((&this->field0.array[((signed int )52u)])));
  tmp__67 =  /*tail*/ pthread_cond_signal(tmp__66);
  tmp__68 =  /*tail*/ pthread_cond_destroy(tmp__66);
  tmp__69 = ((struct l_unnamed32 *)((&this->field0.array[((signed int )4u)])));
  tmp__70 =  /*tail*/ pthread_mutex_unlock(tmp__69);
  tmp__71 =  /*tail*/ pthread_mutex_destroy(tmp__69);
  tmp__72 = ((struct l_unnamed32 *)((&this->field0.array[((signed int )28u)])));
  tmp__73 =  /*tail*/ pthread_mutex_unlock(tmp__72);
  tmp__74 =  /*tail*/ pthread_mutex_destroy(tmp__72);
  tmp__75 =  /*tail*/ pthread_attr_destroy((((struct l_unnamed27 *)((&this->field0.array[((signed int )104u)])))));
  return;
}


static void _ZN7BicycleD0Ev(struct l_class_OC_Bicycle *this) {
  struct l_unnamed26 *tmp__76;
  unsigned int tmp__77;
  unsigned int tmp__78;
  struct l_unnamed32 *tmp__79;
  unsigned int tmp__80;
  unsigned int tmp__81;
  struct l_unnamed32 *tmp__82;
  unsigned int tmp__83;
  unsigned int tmp__84;
  unsigned int tmp__85;

  *(((unsigned int  (***) ( int, ...))this)) = ((unsigned int  (**) ( int, ...))((&_ZTV6Thread.array[((signed int )2u)])));
  tmp__76 = ((struct l_unnamed26 *)((&this->field0.array[((signed int )52u)])));
  tmp__77 =  /*tail*/ pthread_cond_signal(tmp__76);
  tmp__78 =  /*tail*/ pthread_cond_destroy(tmp__76);
  tmp__79 = ((struct l_unnamed32 *)((&this->field0.array[((signed int )4u)])));
  tmp__80 =  /*tail*/ pthread_mutex_unlock(tmp__79);
  tmp__81 =  /*tail*/ pthread_mutex_destroy(tmp__79);
  tmp__82 = ((struct l_unnamed32 *)((&this->field0.array[((signed int )28u)])));
  tmp__83 =  /*tail*/ pthread_mutex_unlock(tmp__82);
  tmp__84 =  /*tail*/ pthread_mutex_destroy(tmp__82);
  tmp__85 =  /*tail*/ pthread_attr_destroy((((struct l_unnamed27 *)((&this->field0.array[((signed int )104u)])))));
   /*tail*/ _ZdlPv(((&this->field0.array[((signed int )0u)])));
  return;
}


static void _GLOBAL__I_a(void) {
  unsigned int tmp__86;

   /*tail*/ _ZNSt8ios_base4InitC1Ev((&_ZStL8__ioinit));
  tmp__86 =  /*tail*/ __cxa_atexit(((void  (*) (unsigned char *))_ZNSt8ios_base4InitD1Ev), ((&_ZStL8__ioinit.field0)), ((unsigned char *)(&__dso_handle)));
  return;
}


static void _ZN7Display3runEv(struct l_class_OC_Display *this) {
  struct l_struct_OC_timespec time_2e_i;    /* Address-exposed local */
  struct l_struct_OC_timespec timeOut_2e_i;    /* Address-exposed local */
  unsigned int *tmp__87;
  struct l_class_OC_std_KD__KD_basic_ofstream *tmp__88;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__89;
  struct l_class_OC_std_KD__KD_basic_string *tmp__90;
  unsigned int *tmp__91;
  unsigned int *tmp__92;
  unsigned int *tmp__93;
  unsigned int *tmp__94;
  struct l_unnamed32 *tmp__95;
  struct l_unnamed26 *tmp__96;
  unsigned int tmp__97;
  unsigned int tmp__98;
  unsigned int tmp__99;
  unsigned int tmp__100;
  unsigned int tmp__101;
  unsigned int tmp__102;
  unsigned int tmp__103;
  unsigned int micro_2e_0_2e_i;
  unsigned int micro_2e_0_2e_i__PHI_TEMPORARY;
  unsigned int tmp__104;
  unsigned int tmp__105;
  unsigned int tmp__106;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__107;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__108;

  tmp__87 = (&this->field3);
  tmp__88 = (&this->field1);
  tmp__89 = ((struct l_class_OC_std_KD__KD_basic_ostream *)tmp__88);
  tmp__90 = (&this->field2);
  tmp__91 = (&time_2e_i.field0);
  tmp__92 = (&timeOut_2e_i.field0);
  tmp__93 = (&time_2e_i.field1);
  tmp__94 = (&timeOut_2e_i.field1);
  tmp__95 = ((struct l_unnamed32 *)((&this->field0.array[((signed int )28u)])));
  tmp__96 = ((struct l_unnamed26 *)((&this->field0.array[((signed int )52u)])));
  goto tmp__109;

  do {     /* Syntactic loop '' to make GCC happy */
tmp__109:
  tmp__97 = *tmp__87;
  tmp__98 = gettimeofday((&time_2e_i), ((struct l_struct_OC_timespec *)/*NULL*/0));
  tmp__99 = *tmp__91;
  tmp__100 = ((unsigned int )(((unsigned int )(((signed int )(((signed int )tmp__97) / ((signed int )1000u))))) + ((unsigned int )tmp__99)));
  *tmp__92 = tmp__100;
  tmp__101 = *tmp__93;
  tmp__102 = ((unsigned int )(((unsigned int )(((unsigned int )(((unsigned int )(((signed int )(((signed int )tmp__97) % ((signed int )1000u))))) * ((unsigned int )1000u))))) + ((unsigned int )tmp__101)));
  if ((((signed int )tmp__102) > ((signed int )1000000u))) {
    goto tmp__110;
  } else {
    micro_2e_0_2e_i__PHI_TEMPORARY = tmp__102;   /* for PHI node */
    goto _ZN6Thread6msleepEl_2e_exit;
  }

_ZN6Thread6msleepEl_2e_exit:
  micro_2e_0_2e_i = micro_2e_0_2e_i__PHI_TEMPORARY;
  *tmp__94 = (((unsigned int )(((unsigned int )micro_2e_0_2e_i) * ((unsigned int )1000u))));
  tmp__104 = pthread_mutex_lock(tmp__95);
  tmp__105 = pthread_cond_timedwait(tmp__96, tmp__95, (&timeOut_2e_i));
  tmp__106 = pthread_mutex_unlock(tmp__95);
  _ZNSt14basic_ofstreamIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode(tmp__88, ((&_OC_str.array[((signed int )0u)])), 17u);
  tmp__107 = _ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(tmp__89, tmp__90);
  tmp__108 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__107, ((&_OC_str1.array[((signed int )0u)])));
  _ZNSt14basic_ofstreamIcSt11char_traitsIcEE5closeEv(tmp__88);
  goto tmp__109;

tmp__110:
  *tmp__92 = (((unsigned int )(((unsigned int )tmp__100) + ((unsigned int )1u))));
  tmp__103 = ((unsigned int )(((unsigned int )tmp__102) + ((unsigned int )4293967296u)));
  micro_2e_0_2e_i__PHI_TEMPORARY = tmp__103;   /* for PHI node */
  goto _ZN6Thread6msleepEl_2e_exit;

  } while (1); /* end of syntactic loop '' */
}


static void _ZN7DisplayD1Ev(struct l_class_OC_Display *this) {
  struct l_unnamed26 *tmp__111;
  unsigned int tmp__112;
  unsigned int tmp__113;
  struct l_unnamed32 *tmp__114;
  unsigned int tmp__115;
  unsigned int tmp__116;
  struct l_unnamed32 *tmp__117;
  unsigned int tmp__118;
  unsigned int tmp__119;
  unsigned int tmp__120;

  *(((unsigned char ***)this)) = ((&_ZTV7Display.array[((signed int )2u)]));
  _ZNSsD1Ev(((&this->field2)));
  _ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev(((&this->field1)));
  *(((unsigned int  (***) ( int, ...))this)) = ((unsigned int  (**) ( int, ...))((&_ZTV6Thread.array[((signed int )2u)])));
  tmp__111 = ((struct l_unnamed26 *)((&this->field0.array[((signed int )52u)])));
  tmp__112 =  /*tail*/ pthread_cond_signal(tmp__111);
  tmp__113 =  /*tail*/ pthread_cond_destroy(tmp__111);
  tmp__114 = ((struct l_unnamed32 *)((&this->field0.array[((signed int )4u)])));
  tmp__115 =  /*tail*/ pthread_mutex_unlock(tmp__114);
  tmp__116 =  /*tail*/ pthread_mutex_destroy(tmp__114);
  tmp__117 = ((struct l_unnamed32 *)((&this->field0.array[((signed int )28u)])));
  tmp__118 =  /*tail*/ pthread_mutex_unlock(tmp__117);
  tmp__119 =  /*tail*/ pthread_mutex_destroy(tmp__117);
  tmp__120 =  /*tail*/ pthread_attr_destroy((((struct l_unnamed27 *)((&this->field0.array[((signed int )104u)])))));
  return;
}


static void _ZN7DisplayD0Ev(struct l_class_OC_Display *this) {
  _ZN7DisplayD1Ev(this);
   /*tail*/ _ZdlPv(((&this->field0.array[((signed int )0u)])));
  return;
}


static void _GLOBAL__I_a7(void) {
  unsigned int tmp__121;

   /*tail*/ _ZNSt8ios_base4InitC1Ev((&_ZStL8__ioinit1));
  tmp__121 =  /*tail*/ __cxa_atexit(((void  (*) (unsigned char *))_ZNSt8ios_base4InitD1Ev), ((&_ZStL8__ioinit1.field0)), ((unsigned char *)(&__dso_handle)));
  return;
}


unsigned int main(void) {
  struct l_class_OC_std_KD__KD_basic_stringstream msg_2e_i;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string tmp__122;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_allocator tmp__123;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string tmp__124;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__125;
  unsigned char *tmp__126;
  float *tmp__127;
  unsigned char *tmp__128;
  unsigned int tmp__129;
  unsigned long long tmp_2e_i;
  unsigned int tmp__130;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__131;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__132;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__133;
  float tmp__134;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__135;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__136;
  unsigned char *tmp__137;
  unsigned int tmp__138;
  unsigned int tmp__139;
  unsigned int tmp__140;
  struct l_unnamed27 *tmp__141;
  unsigned int tmp__142;
  unsigned int tmp__143;
  unsigned int tmp__144;
  struct l_class_OC_Display **tmp__145;
  struct l_class_OC_Display *tmp__146;
  unsigned int tmp__147;
  struct l_class_OC_Display *tmp__148;
  struct l_class_OC_Display *tmp__149;
  void  (**tmp__150) (struct l_class_OC_Thread *);
  void  (*tmp__151) (struct l_class_OC_Thread *);
  struct l_class_OC_Display *tmp__152;
  unsigned int tmp__153;
  unsigned char *tmp__154;
  unsigned int tmp__155;
  unsigned int tmp__156;
  unsigned int tmp__157;
  struct l_unnamed27 *tmp__158;
  unsigned int tmp__159;
  unsigned int tmp__160;
  unsigned int tmp__161;
  unsigned int *tmp__162;
  unsigned int tmp__163;
  unsigned int tmp__164;
  unsigned int tmp__165;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__166;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__167;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__168;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__169;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__170;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__171;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__172;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__173;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__174;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__175;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__176;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__177;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__178;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__179;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__180;
  struct l_class_OC_std_KD__KD_basic_ostream *tmp__181;

  CODE_FOR_MAIN();
  tmp__125 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str14.array[((signed int )0u)])));
  tmp__126 = _Znwj(64u);
  *(((unsigned int *)((&tmp__126[((signed int )24u)])))) = 0u;
  *((&tmp__126[((signed int )28u)])) = ((unsigned char )0);
  tmp__127 = ((float *)((&tmp__126[((signed int )48u)])));
  *tmp__127 = 0.000000;
  *(((float *)((&tmp__126[((signed int )52u)])))) = 0.000000;
  *(((float *)((&tmp__126[((signed int )56u)])))) = 0.000000;
  tmp__128 = (&tmp__126[((signed int )40u)]);
  tmp__129 = gettimeofday((((struct l_struct_OC_timespec *)tmp__128)), ((struct l_struct_OC_timespec *)/*NULL*/0));
  tmp_2e_i = *(((unsigned long long *)tmp__128));
  *(((unsigned long long *)((&tmp__126[((signed int )32u)])))) = tmp_2e_i;
  tmp__130 = pthread_mutex_init((((struct l_unnamed32 *)tmp__126)), ((struct l_union_OC_anon *)/*NULL*/0));
  _ZNSt18basic_stringstreamIcSt11char_traitsIcESaIcEEC1ESt13_Ios_Openmode((&msg_2e_i), 24u);
  _ZNSaIcEC1Ev((&tmp__123));
  _ZNSsC1EPKcRKSaIcE((&tmp__122), ((&_OC_str2.array[((signed int )0u)])), (&tmp__123));
  tmp__131 = ((struct l_class_OC_std_KD__KD_basic_ostream *)((&msg_2e_i.field0.array[((signed int )8u)])));
  tmp__132 = _ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(tmp__131, (&tmp__122));
  _ZNSsD1Ev((&tmp__122));
  _ZNSaIcED1Ev((&tmp__123));
  tmp__133 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__131, ((&_OC_str13.array[((signed int )0u)])));
  tmp__134 = *tmp__127;
  tmp__135 = _ZNSolsEf(tmp__133, tmp__134);
  tmp__136 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(tmp__135, ((&_OC_str24.array[((signed int )0u)])));
  tmp__137 = _Znwj(428u);
  tmp__124 = _ZNKSt18basic_stringstreamIcSt11char_traitsIcESaIcEE3strEv((&msg_2e_i));
  *(((unsigned int  (***) ( int, ...))tmp__137)) = ((unsigned int  (**) ( int, ...))((&_ZTV6Thread.array[((signed int )2u)])));
  tmp__138 = pthread_mutex_init((((struct l_unnamed32 *)((&tmp__137[((signed int )4u)])))), ((struct l_union_OC_anon *)/*NULL*/0));
  tmp__139 = pthread_mutex_init((((struct l_unnamed32 *)((&tmp__137[((signed int )28u)])))), ((struct l_union_OC_anon *)/*NULL*/0));
  *((&tmp__137[((signed int )100u)])) = ((unsigned char )0);
  tmp__140 = pthread_cond_init((((struct l_unnamed26 *)((&tmp__137[((signed int )52u)])))), ((struct l_union_OC_anon *)/*NULL*/0));
  tmp__141 = ((struct l_unnamed27 *)((&tmp__137[((signed int )104u)])));
  tmp__142 = pthread_attr_init(tmp__141);
  tmp__143 = pthread_attr_setdetachstate(tmp__141, 1u);
  tmp__144 = pthread_attr_setscope(tmp__141, 0u);
  *(((unsigned char ***)tmp__137)) = ((&_ZTV7Display.array[((signed int )2u)]));
  _ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1Ev((((struct l_class_OC_std_KD__KD_basic_ofstream *)((&tmp__137[((signed int )144u)])))));
  _ZNSsC1ERKSs((((struct l_class_OC_std_KD__KD_basic_string *)((&tmp__137[((signed int )420u)])))), (&tmp__124));
  *(((unsigned int *)((&tmp__137[((signed int )424u)])))) = 0u;
  tmp__145 = ((struct l_class_OC_Display **)((&tmp__126[((signed int )60u)])));
  *tmp__145 = (((struct l_class_OC_Display *)tmp__137));
  _ZNSsD1Ev((&tmp__124));
  tmp__146 = *tmp__145;
  tmp__147 = pthread_mutex_lock((((struct l_unnamed32 *)((&tmp__146->field0.array[((signed int )4u)])))));
  tmp__148 = *tmp__145;
  *((&tmp__148->field3)) = 200u;
  tmp__149 = *tmp__145;
  tmp__150 = *(((void  (***) (struct l_class_OC_Thread *))tmp__149));
  tmp__151 = *((&tmp__150[((signed int )3u)]));
  tmp__151((((struct l_class_OC_Thread *)tmp__149)));
  tmp__152 = *tmp__145;
  tmp__153 = pthread_mutex_unlock((((struct l_unnamed32 *)((&tmp__152->field0.array[((signed int )4u)])))));
  _ZNSt18basic_stringstreamIcSt11char_traitsIcESaIcEED1Ev((&msg_2e_i));
  tmp__154 = _Znwj(148u);
  *(((unsigned int  (***) ( int, ...))tmp__154)) = ((unsigned int  (**) ( int, ...))((&_ZTV6Thread.array[((signed int )2u)])));
  tmp__155 = pthread_mutex_init((((struct l_unnamed32 *)((&tmp__154[((signed int )4u)])))), ((struct l_union_OC_anon *)/*NULL*/0));
  tmp__156 = pthread_mutex_init((((struct l_unnamed32 *)((&tmp__154[((signed int )28u)])))), ((struct l_union_OC_anon *)/*NULL*/0));
  *((&tmp__154[((signed int )100u)])) = ((unsigned char )0);
  tmp__157 = pthread_cond_init((((struct l_unnamed26 *)((&tmp__154[((signed int )52u)])))), ((struct l_union_OC_anon *)/*NULL*/0));
  tmp__158 = ((struct l_unnamed27 *)((&tmp__154[((signed int )104u)])));
  tmp__159 = pthread_attr_init(tmp__158);
  tmp__160 = pthread_attr_setdetachstate(tmp__158, 1u);
  tmp__161 = pthread_attr_setscope(tmp__158, 0u);
  *(((unsigned char ***)tmp__154)) = ((&_ZTV7Bicycle.array[((signed int )2u)]));
  *(((struct l_class_OC_EmbeddedPC **)((&tmp__154[((signed int )144u)])))) = (((struct l_class_OC_EmbeddedPC *)tmp__126));
  tmp__162 = ((unsigned int *)((&tmp__154[((signed int )140u)])));
  tmp__163 = pthread_create(tmp__162, tmp__158, _ZN6Thread8functionEPv, tmp__154);
  tmp__164 = *tmp__162;
  tmp__165 = pthread_detach(tmp__164);
  tmp__166 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str115.array[((signed int )0u)])));
  tmp__167 = _ZNSolsEPFRSoS_E(tmp__166, _ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_);
  tmp__168 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str216.array[((signed int )0u)])));
  tmp__169 = _ZNSolsEPFRSoS_E(tmp__168, _ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_);
  tmp__170 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str317.array[((signed int )0u)])));
  tmp__171 = _ZNSolsEPFRSoS_E(tmp__170, _ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_);
  tmp__172 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str418.array[((signed int )0u)])));
  tmp__173 = _ZNSolsEPFRSoS_E(tmp__172, _ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_);
  tmp__174 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str519.array[((signed int )0u)])));
  tmp__175 = _ZNSolsEPFRSoS_E(tmp__174, _ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_);
  tmp__176 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str620.array[((signed int )0u)])));
  tmp__177 = _ZNSolsEPFRSoS_E(tmp__176, _ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_);
  tmp__178 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str721.array[((signed int )0u)])));
  tmp__179 = _ZNSolsEPFRSoS_E(tmp__178, _ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_);
  tmp__180 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str216.array[((signed int )0u)])));
  tmp__181 = _ZNSolsEPFRSoS_E(tmp__180, _ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_);
  __assert_fail(((&_OC_str822.array[((signed int )0u)])), ((&_OC_str923.array[((signed int )0u)])), 39u, ((&__PRETTY_FUNCTION___OC_main.array[((signed int )0u)])));
  /*UNREACHABLE*/;
}


static void _GLOBAL__I_a25(void) {
  unsigned int tmp__182;

   /*tail*/ _ZNSt8ios_base4InitC1Ev((&_ZStL8__ioinit8));
  tmp__182 =  /*tail*/ __cxa_atexit(((void  (*) (unsigned char *))_ZNSt8ios_base4InitD1Ev), ((&_ZStL8__ioinit8.field0)), ((unsigned char *)(&__dso_handle)));
  return;
}


static void _ZN6ThreadD0Ev(struct l_class_OC_Thread *this) {
  struct l_unnamed26 *tmp__183;
  unsigned int tmp__184;
  unsigned int tmp__185;
  struct l_unnamed32 *tmp__186;
  unsigned int tmp__187;
  unsigned int tmp__188;
  struct l_unnamed32 *tmp__189;
  unsigned int tmp__190;
  unsigned int tmp__191;
  unsigned int tmp__192;

  *((&this->field0)) = ((unsigned int  (**) ( int, ...))((&_ZTV6Thread.array[((signed int )2u)])));
  tmp__183 = (&this->field3);
  tmp__184 =  /*tail*/ pthread_cond_signal(tmp__183);
  tmp__185 =  /*tail*/ pthread_cond_destroy(tmp__183);
  tmp__186 = (&this->field1);
  tmp__187 =  /*tail*/ pthread_mutex_unlock(tmp__186);
  tmp__188 =  /*tail*/ pthread_mutex_destroy(tmp__186);
  tmp__189 = (&this->field2);
  tmp__190 =  /*tail*/ pthread_mutex_unlock(tmp__189);
  tmp__191 =  /*tail*/ pthread_mutex_destroy(tmp__189);
  tmp__192 =  /*tail*/ pthread_attr_destroy(((&this->field5)));
   /*tail*/ _ZdlPv((((unsigned char *)this)));
  return;
}


static void _ZN6ThreadD2Ev(struct l_class_OC_Thread *this) {
  struct l_unnamed26 *tmp__193;
  unsigned int tmp__194;
  unsigned int tmp__195;
  struct l_unnamed32 *tmp__196;
  unsigned int tmp__197;
  unsigned int tmp__198;
  struct l_unnamed32 *tmp__199;
  unsigned int tmp__200;
  unsigned int tmp__201;
  unsigned int tmp__202;

  *((&this->field0)) = ((unsigned int  (**) ( int, ...))((&_ZTV6Thread.array[((signed int )2u)])));
  tmp__193 = (&this->field3);
  tmp__194 =  /*tail*/ pthread_cond_signal(tmp__193);
  tmp__195 =  /*tail*/ pthread_cond_destroy(tmp__193);
  tmp__196 = (&this->field1);
  tmp__197 =  /*tail*/ pthread_mutex_unlock(tmp__196);
  tmp__198 =  /*tail*/ pthread_mutex_destroy(tmp__196);
  tmp__199 = (&this->field2);
  tmp__200 =  /*tail*/ pthread_mutex_unlock(tmp__199);
  tmp__201 =  /*tail*/ pthread_mutex_destroy(tmp__199);
  tmp__202 =  /*tail*/ pthread_attr_destroy(((&this->field5)));
  return;
}


static unsigned char *_ZN6Thread8functionEPv(unsigned char *ptr) {
  void  (**tmp__203) (struct l_class_OC_Thread *);
  void  (*tmp__204) (struct l_class_OC_Thread *);

  if ((ptr == ((unsigned char *)/*NULL*/0))) {
    goto tmp__205;
  } else {
    goto tmp__206;
  }

tmp__205:
  return ((unsigned char *)/*NULL*/0);
tmp__206:
  tmp__203 = *(((void  (***) (struct l_class_OC_Thread *))ptr));
  tmp__204 = *tmp__203;
   /*tail*/ tmp__204((((struct l_class_OC_Thread *)ptr)));
   /*tail*/ pthread_exit(ptr);
  /*UNREACHABLE*/;
}


static void _ZN6Thread5startEv(struct l_class_OC_Thread *this) {
  unsigned int *tmp__207;
  unsigned int tmp__208;
  unsigned int tmp__209;
  unsigned int tmp__210;

  tmp__207 = (&this->field6);
  tmp__208 =  /*tail*/ pthread_create(tmp__207, ((&this->field5)), _ZN6Thread8functionEPv, (((unsigned char *)this)));
  tmp__209 = *tmp__207;
  tmp__210 =  /*tail*/ pthread_detach(tmp__209);
  return;
}

/**********************VarMap.dat (Number Line) Content************************/
/******************************************************************************/

/**********************VarMap.dat (Variable Name) Content************************/
/******************************************************************************/