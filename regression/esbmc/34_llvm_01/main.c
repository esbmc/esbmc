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
struct l_class_OC_std_KD__KD_allocator;
struct l_class_OC_std_KD__KD_basic_ostream;
struct l_class_OC_std_KD__KD_basic_string;
struct l_struct_OC_std_KD__KD_basic_string_MD_char_MC__AC_std_KD__KD_char_traits_MD_char_OD__MC__AC_std_KD__KD_allocator_MD_char_OD__AC__OD__KD__KD__Alloc_hider;
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
struct l_unnamed3;
struct l_unnamed4;
struct l_unnamed5;
struct l_unnamed6;
struct l_unnamed7;
struct l_unnamed8;
struct l_unnamed9;

/* Typedefs */
typedef struct l_class_OC_std_KD__KD_allocator l_class_OC_std_KD__KD_allocator;
typedef struct l_class_OC_std_KD__KD_basic_ostream l_class_OC_std_KD__KD_basic_ostream;
typedef struct l_class_OC_std_KD__KD_basic_string l_class_OC_std_KD__KD_basic_string;
typedef struct l_struct_OC_std_KD__KD_basic_string_MD_char_MC__AC_std_KD__KD_char_traits_MD_char_OD__MC__AC_std_KD__KD_allocator_MD_char_OD__AC__OD__KD__KD__Alloc_hider l_struct_OC_std_KD__KD_basic_string_MD_char_MC__AC_std_KD__KD_char_traits_MD_char_OD__MC__AC_std_KD__KD_allocator_MD_char_OD__AC__OD__KD__KD__Alloc_hider;
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
typedef struct l_unnamed3 l_unnamed3;
typedef struct l_unnamed4 l_unnamed4;
typedef struct l_unnamed5 l_unnamed5;
typedef struct l_unnamed6 l_unnamed6;
typedef struct l_unnamed7 l_unnamed7;
typedef struct l_unnamed8 l_unnamed8;
typedef struct l_unnamed9 l_unnamed9;

/* Structure contents */
struct l_class_OC_std_KD__KD_allocator {
  unsigned char field0;
};

struct l_unnamed19 { unsigned char array[136]; };

struct l_class_OC_std_KD__KD_basic_ostream {
  unsigned int  (**field0) ( int, ...);
  struct l_unnamed19 field1;
};

struct l_struct_OC_std_KD__KD_basic_string_MD_char_MC__AC_std_KD__KD_char_traits_MD_char_OD__MC__AC_std_KD__KD_allocator_MD_char_OD__AC__OD__KD__KD__Alloc_hider {
  unsigned char *field0;
};

struct l_class_OC_std_KD__KD_basic_string {
  struct l_struct_OC_std_KD__KD_basic_string_MD_char_MC__AC_std_KD__KD_char_traits_MD_char_OD__MC__AC_std_KD__KD_allocator_MD_char_OD__AC__OD__KD__KD__Alloc_hider field0;
};

struct l_unnamed0 { unsigned char array[24]; };

struct l_unnamed1 { unsigned char array[26]; };

struct l_unnamed10 { unsigned char array[45]; };

struct l_unnamed12 {
  unsigned int field0;
  void  (*field1) (void);
};

struct l_unnamed11 { struct l_unnamed12 array[1]; };

struct l_unnamed13 { unsigned char array[33]; };

struct l_unnamed14 { unsigned char array[38]; };

struct l_unnamed15 { unsigned char array[10]; };

struct l_unnamed16 { unsigned char array[11]; };

struct l_unnamed17 { unsigned char array[8]; };

struct l_unnamed18 { unsigned char array[6]; };

struct l_unnamed2 { unsigned char array[3]; };

struct l_unnamed20 { unsigned char array[18]; };

struct l_unnamed3 { unsigned char array[2]; };

struct l_unnamed4 { unsigned char array[48]; };

struct l_unnamed5 { unsigned char array[34]; };

struct l_unnamed6 { unsigned char array[28]; };

struct l_unnamed7 { unsigned char array[23]; };

struct l_unnamed8 { unsigned char array[43]; };

struct l_unnamed9 { unsigned char array[5]; };


/* External Global Variable Declarations */
extern unsigned char *__dso_handle;
extern struct l_class_OC_std_KD__KD_basic_ostream _ZSt4cout;

/* Function Declarations */
double fmod(double, double);
float fmodf(float, float);
long double fmodl(long double, long double);
static void __cxx_global_var_init(void);
void _ZNSt8ios_base4InitC1Ev(struct l_class_OC_std_KD__KD_allocator *);
void _ZNSt8ios_base4InitD1Ev(struct l_class_OC_std_KD__KD_allocator *);
unsigned int __cxa_atexit(void  (*) (unsigned char *), unsigned char *, unsigned char *);
unsigned int main(void);
void _ZNSsC1EPKcRKSaIcE(struct l_class_OC_std_KD__KD_basic_string *, unsigned char *, struct l_class_OC_std_KD__KD_allocator *);
void _ZNSaIcEC1Ev(struct l_class_OC_std_KD__KD_allocator *);
unsigned int __gxx_personality_v0(int vararg_dummy_arg,...);
void _Unwind_Resume_or_Rethrow(unsigned char *);
void _ZNSaIcED1Ev(struct l_class_OC_std_KD__KD_allocator *);
void _ZNSsC1Ev(struct l_class_OC_std_KD__KD_basic_string *);
struct l_class_OC_std_KD__KD_basic_ostream *_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(struct l_class_OC_std_KD__KD_basic_ostream *, unsigned char *);
struct l_class_OC_std_KD__KD_basic_ostream *_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(struct l_class_OC_std_KD__KD_basic_ostream *, signed char );
struct l_class_OC_std_KD__KD_basic_ostream *_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(struct l_class_OC_std_KD__KD_basic_ostream *, struct l_class_OC_std_KD__KD_basic_string *);
bool _ZSteqIcEN9__gnu_cxx11__enable_ifIXsrSt9__is_charIT_E7__valueEbE6__typeERKSbIS3_St11char_traitsIS3_ESaIS3_EESC_(struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___lhs, struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___rhs) __ATTRIBUTE_WEAK__;
bool _ZStneIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_(struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___lhs, struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___rhs) __ATTRIBUTE_WEAK__;
bool _ZStgtIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_(struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___lhs, struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___rhs) __ATTRIBUTE_WEAK__;
bool _ZStltIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_(struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___lhs, struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___rhs) __ATTRIBUTE_WEAK__;
bool _ZStgeIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_(struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___lhs, struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___rhs) __ATTRIBUTE_WEAK__;
bool _ZStleIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_(struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___lhs, struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___rhs) __ATTRIBUTE_WEAK__;
bool _ZNKSs5emptyEv(struct l_class_OC_std_KD__KD_basic_string *);
struct l_class_OC_std_KD__KD_basic_string *_ZNSsaSERKSs(struct l_class_OC_std_KD__KD_basic_string *, struct l_class_OC_std_KD__KD_basic_string *);
struct l_class_OC_std_KD__KD_basic_string *_ZNSspLERKSs(struct l_class_OC_std_KD__KD_basic_string *, struct l_class_OC_std_KD__KD_basic_string *);
struct l_class_OC_std_KD__KD_basic_string *_ZNSspLEPKc(struct l_class_OC_std_KD__KD_basic_string *, unsigned char *);
struct l_class_OC_std_KD__KD_basic_string _ZNKSs6substrEjj(struct l_class_OC_std_KD__KD_basic_string *, unsigned int , unsigned int );
void _ZNSsD1Ev(struct l_class_OC_std_KD__KD_basic_string *);
void _ZSt9terminatev(void);
unsigned char *_Znwj(unsigned int );
void _ZNSsC1ERKSs(struct l_class_OC_std_KD__KD_basic_string *, struct l_class_OC_std_KD__KD_basic_string *);
void _ZdlPv(unsigned char *);
unsigned char *_ZNSsixEj(struct l_class_OC_std_KD__KD_basic_string *, unsigned int );
struct l_class_OC_std_KD__KD_basic_ostream *_ZNSolsEPFRSoS_E(struct l_class_OC_std_KD__KD_basic_ostream *, struct l_class_OC_std_KD__KD_basic_ostream * (*) (struct l_class_OC_std_KD__KD_basic_ostream *));
struct l_class_OC_std_KD__KD_basic_ostream *_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(struct l_class_OC_std_KD__KD_basic_ostream *);
unsigned char *_ZNSs2atEj(struct l_class_OC_std_KD__KD_basic_string *, unsigned int );
unsigned int _ZNKSs7compareERKSs(struct l_class_OC_std_KD__KD_basic_string *, struct l_class_OC_std_KD__KD_basic_string *);
unsigned int _ZNKSs4sizeEv(struct l_class_OC_std_KD__KD_basic_string *);
unsigned int _ZNSt11char_traitsIcE7compareEPKcS2_j(unsigned char *llvm_cbe___s1, unsigned char *llvm_cbe___s2, unsigned int llvm_cbe___n) __ATTRIBUTE_WEAK__;
unsigned char *_ZNKSs4dataEv(struct l_class_OC_std_KD__KD_basic_string *);
unsigned int memcmp(unsigned char *, unsigned char *, unsigned int );
static void _GLOBAL__I_a(void) __ATTRIBUTE_CTOR__;
void abort(void);


/* Global Variable Declarations */
static struct l_class_OC_std_KD__KD_allocator _ZStL8__ioinit;
static struct l_unnamed18 _OC_str;
static struct l_unnamed15 _OC_str1;
static struct l_unnamed17 _OC_str2;
static struct l_unnamed16 _OC_str3;
static struct l_unnamed16 _OC_str4;
static struct l_unnamed14 _OC_str5;
static struct l_unnamed20 _OC_str6;
static struct l_unnamed9 _OC_str7;
static struct l_unnamed18 _OC_str8;
static struct l_unnamed20 _OC_str9;
static struct l_unnamed20 _OC_str10;
static struct l_unnamed20 _OC_str11;
static struct l_unnamed20 _OC_str12;
static struct l_unnamed20 _OC_str13;
static struct l_unnamed7 _OC_str14;
static struct l_unnamed5 _OC_str15;
static struct l_unnamed17 _OC_str16;
static struct l_unnamed3 _OC_str17;
static struct l_unnamed0 _OC_str18;
static struct l_unnamed1 _OC_str19;
static struct l_unnamed17 _OC_str20;
static struct l_unnamed18 _OC_str21;
static struct l_unnamed2 _OC_str22;
static struct l_unnamed4 _OC_str23;
static struct l_unnamed14 _OC_str24;
static struct l_unnamed13 _OC_str25;
static struct l_unnamed13 _OC_str26;
static struct l_unnamed16 _OC_str27;
static struct l_unnamed6 _OC_str28;
static struct l_unnamed15 _OC_str29;
static struct l_unnamed8 _OC_str30;
static struct l_unnamed10 _OC_str31;


/* Global Variable Definitions and Initialization */
static struct l_class_OC_std_KD__KD_allocator _ZStL8__ioinit;
static struct l_unnamed18 _OC_str = { "happy" };
static struct l_unnamed15 _OC_str1 = { " birthday" };
static struct l_unnamed17 _OC_str2 = { "s1 is \"" };
static struct l_unnamed16 _OC_str3 = { "\"; s2 is \"" };
static struct l_unnamed16 _OC_str4 = { "\"; s3 is \"" };
static struct l_unnamed14 _OC_str5 = { "\n\nThe results of comparing s2 and s1:" };
static struct l_unnamed20 _OC_str6 = { "\ns2 == s1 yields " };
static struct l_unnamed9 _OC_str7 = { "true" };
static struct l_unnamed18 _OC_str8 = { "false" };
static struct l_unnamed20 _OC_str9 = { "\ns2 != s1 yields " };
static struct l_unnamed20 _OC_str10 = { "\ns2 >  s1 yields " };
static struct l_unnamed20 _OC_str11 = { "\ns2 <  s1 yields " };
static struct l_unnamed20 _OC_str12 = { "\ns2 >= s1 yields " };
static struct l_unnamed20 _OC_str13 = { "\ns2 <= s1 yields " };
static struct l_unnamed7 _OC_str14 = { "\n\nTesting s3.empty():\n" };
static struct l_unnamed5 _OC_str15 = { "s3 is empty; assigning s1 to s3;\n" };
static struct l_unnamed17 _OC_str16 = { "s3 is \"" };
static struct l_unnamed3 _OC_str17 = { "\"" };
static struct l_unnamed0 _OC_str18 = { "\n\ns1 += s2 yields s1 = " };
static struct l_unnamed1 _OC_str19 = { "\n\ns1 += \" to you\" yields\n" };
static struct l_unnamed17 _OC_str20 = { " to you" };
static struct l_unnamed18 _OC_str21 = { "s1 = " };
static struct l_unnamed2 _OC_str22 = { "\n\n" };
static struct l_unnamed4 _OC_str23 = { "The substring of s1 starting at location 0 for\n" };
static struct l_unnamed14 _OC_str24 = { "14 characters, s1.substr(0, 14), is:\n" };
static struct l_unnamed13 _OC_str25 = { "The substring of s1 starting at\n" };
static struct l_unnamed13 _OC_str26 = { "location 15, s1.substr(15), is:\n" };
static struct l_unnamed16 _OC_str27 = { "\n*s4Ptr = " };
static struct l_unnamed6 _OC_str28 = { "assigning *s4Ptr to *s4Ptr\n" };
static struct l_unnamed15 _OC_str29 = { "*s4Ptr = " };
static struct l_unnamed8 _OC_str30 = { "\ns1 after s1[0] = 'H' and s1[6] = 'B' is: " };
static struct l_unnamed10 _OC_str31 = { "Attempt to assign 'd' to s1.at( 30 ) yields:" };


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

static void __cxx_global_var_init(void) {
  unsigned int llvm_cbe_tmp__1;

  _ZNSt8ios_base4InitC1Ev((&_ZStL8__ioinit));
  llvm_cbe_tmp__1 = __cxa_atexit(((void  (*) (unsigned char *))_ZNSt8ios_base4InitD1Ev), ((&_ZStL8__ioinit.field0)), ((unsigned char *)(&__dso_handle)));
  return;
}


unsigned int main(void) {
  unsigned int llvm_cbe_tmp__2;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string llvm_cbe_s1;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_allocator llvm_cbe_tmp__3;    /* Address-exposed local */
  unsigned char *llvm_cbe_tmp__4;    /* Address-exposed local */
  unsigned int llvm_cbe_tmp__5;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string llvm_cbe_s2;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_allocator llvm_cbe_tmp__6;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string llvm_cbe_s3;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string llvm_cbe_tmp__7;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string llvm_cbe_tmp__8;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_s4Ptr;    /* Address-exposed local */
  unsigned int llvm_cbe_tmp__9;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__10;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__11;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__12;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__13;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__14;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__15;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__16;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__17;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__18;
  bool llvm_cbe_tmp__19;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__20;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__21;
  bool llvm_cbe_tmp__22;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__23;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__24;
  bool llvm_cbe_tmp__25;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__26;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__27;
  bool llvm_cbe_tmp__28;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__29;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__30;
  bool llvm_cbe_tmp__31;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__32;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__33;
  bool llvm_cbe_tmp__34;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__35;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__36;
  bool llvm_cbe_tmp__37;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__38;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__39;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__40;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__41;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__42;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__43;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__44;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__45;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__46;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__47;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__48;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__49;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__50;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__51;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__52;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__53;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__54;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__55;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__56;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__57;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__58;
  unsigned char *llvm_cbe_tmp__59;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__60;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__61;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__62;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__63;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__64;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__65;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__66;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__67;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__68;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__69;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__70;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__71;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__72;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__73;
  unsigned char *llvm_cbe_tmp__74;
  unsigned char *llvm_cbe_tmp__75;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__76;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__77;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__78;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__79;
  struct l_class_OC_std_KD__KD_basic_ostream *llvm_cbe_tmp__80;
  unsigned char *llvm_cbe_tmp__81;
  unsigned int llvm_cbe_tmp__82;

  CODE_FOR_MAIN();
  *(&llvm_cbe_tmp__2) = 0u;
  _ZNSaIcEC1Ev((&llvm_cbe_tmp__3));
  _ZNSsC1EPKcRKSaIcE((&llvm_cbe_s1), ((&_OC_str.array[((signed int )0u)])), (&llvm_cbe_tmp__3));
  _ZNSaIcED1Ev((&llvm_cbe_tmp__3));
  _ZNSaIcEC1Ev((&llvm_cbe_tmp__6));
  _ZNSsC1EPKcRKSaIcE((&llvm_cbe_s2), ((&_OC_str1.array[((signed int )0u)])), (&llvm_cbe_tmp__6));
  _ZNSaIcED1Ev((&llvm_cbe_tmp__6));
  _ZNSsC1Ev((&llvm_cbe_s3));
  llvm_cbe_tmp__10 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str2.array[((signed int )0u)])));
  llvm_cbe_tmp__11 = _ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(llvm_cbe_tmp__10, (&llvm_cbe_s1));
  llvm_cbe_tmp__12 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__11, ((&_OC_str3.array[((signed int )0u)])));
  llvm_cbe_tmp__13 = _ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(llvm_cbe_tmp__12, (&llvm_cbe_s2));
  llvm_cbe_tmp__14 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__13, ((&_OC_str4.array[((signed int )0u)])));
  llvm_cbe_tmp__15 = _ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(llvm_cbe_tmp__14, (&llvm_cbe_s3));
  llvm_cbe_tmp__16 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(llvm_cbe_tmp__15, ((unsigned char )34));
  llvm_cbe_tmp__17 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__16, ((&_OC_str5.array[((signed int )0u)])));
  llvm_cbe_tmp__18 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__17, ((&_OC_str6.array[((signed int )0u)])));
  llvm_cbe_tmp__19 = ((_ZSteqIcEN9__gnu_cxx11__enable_ifIXsrSt9__is_charIT_E7__valueEbE6__typeERKSbIS3_St11char_traitsIS3_ESaIS3_EESC_((&llvm_cbe_s2), (&llvm_cbe_s1)))&1);
  llvm_cbe_tmp__20 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__18, (((llvm_cbe_tmp__19) ? (((&_OC_str7.array[((signed int )0u)]))) : (((&_OC_str8.array[((signed int )0u)]))))));
  llvm_cbe_tmp__21 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__20, ((&_OC_str9.array[((signed int )0u)])));
  llvm_cbe_tmp__22 = ((_ZStneIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_((&llvm_cbe_s2), (&llvm_cbe_s1)))&1);
  llvm_cbe_tmp__23 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__21, (((llvm_cbe_tmp__22) ? (((&_OC_str7.array[((signed int )0u)]))) : (((&_OC_str8.array[((signed int )0u)]))))));
  llvm_cbe_tmp__24 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__23, ((&_OC_str10.array[((signed int )0u)])));
  llvm_cbe_tmp__25 = ((_ZStgtIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_((&llvm_cbe_s2), (&llvm_cbe_s1)))&1);
  llvm_cbe_tmp__26 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__24, (((llvm_cbe_tmp__25) ? (((&_OC_str7.array[((signed int )0u)]))) : (((&_OC_str8.array[((signed int )0u)]))))));
  llvm_cbe_tmp__27 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__26, ((&_OC_str11.array[((signed int )0u)])));
  llvm_cbe_tmp__28 = ((_ZStltIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_((&llvm_cbe_s2), (&llvm_cbe_s1)))&1);
  llvm_cbe_tmp__29 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__27, (((llvm_cbe_tmp__28) ? (((&_OC_str7.array[((signed int )0u)]))) : (((&_OC_str8.array[((signed int )0u)]))))));
  llvm_cbe_tmp__30 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__29, ((&_OC_str12.array[((signed int )0u)])));
  llvm_cbe_tmp__31 = ((_ZStgeIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_((&llvm_cbe_s2), (&llvm_cbe_s1)))&1);
  llvm_cbe_tmp__32 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__30, (((llvm_cbe_tmp__31) ? (((&_OC_str7.array[((signed int )0u)]))) : (((&_OC_str8.array[((signed int )0u)]))))));
  llvm_cbe_tmp__33 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__32, ((&_OC_str13.array[((signed int )0u)])));
  llvm_cbe_tmp__34 = ((_ZStleIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_((&llvm_cbe_s2), (&llvm_cbe_s1)))&1);
  llvm_cbe_tmp__35 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__33, (((llvm_cbe_tmp__34) ? (((&_OC_str7.array[((signed int )0u)]))) : (((&_OC_str8.array[((signed int )0u)]))))));
  llvm_cbe_tmp__36 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str14.array[((signed int )0u)])));
  llvm_cbe_tmp__37 = ((_ZNKSs5emptyEv((&llvm_cbe_s3)))&1);
  if (llvm_cbe_tmp__37) {
    goto llvm_cbe_tmp__83;
  } else {
    goto llvm_cbe_tmp__84;
  }

llvm_cbe_tmp__83:
  llvm_cbe_tmp__38 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str15.array[((signed int )0u)])));
  llvm_cbe_tmp__39 = _ZNSsaSERKSs((&llvm_cbe_s3), (&llvm_cbe_s1));
  llvm_cbe_tmp__40 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str16.array[((signed int )0u)])));
  llvm_cbe_tmp__41 = _ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(llvm_cbe_tmp__40, (&llvm_cbe_s3));
  llvm_cbe_tmp__42 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__41, ((&_OC_str17.array[((signed int )0u)])));
  goto llvm_cbe_tmp__84;

llvm_cbe_tmp__84:
  llvm_cbe_tmp__43 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str18.array[((signed int )0u)])));
  llvm_cbe_tmp__44 = _ZNSspLERKSs((&llvm_cbe_s1), (&llvm_cbe_s2));
  llvm_cbe_tmp__45 = _ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E((&_ZSt4cout), (&llvm_cbe_s1));
  llvm_cbe_tmp__46 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str19.array[((signed int )0u)])));
  llvm_cbe_tmp__47 = _ZNSspLEPKc((&llvm_cbe_s1), ((&_OC_str20.array[((signed int )0u)])));
  llvm_cbe_tmp__48 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str21.array[((signed int )0u)])));
  llvm_cbe_tmp__49 = _ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(llvm_cbe_tmp__48, (&llvm_cbe_s1));
  llvm_cbe_tmp__50 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__49, ((&_OC_str22.array[((signed int )0u)])));
  llvm_cbe_tmp__51 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str23.array[((signed int )0u)])));
  llvm_cbe_tmp__52 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__51, ((&_OC_str24.array[((signed int )0u)])));
  llvm_cbe_tmp__7 = _ZNKSs6substrEjj((&llvm_cbe_s1), 0u, 14u);
  llvm_cbe_tmp__53 = _ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(llvm_cbe_tmp__52, (&llvm_cbe_tmp__7));
  llvm_cbe_tmp__54 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__53, ((&_OC_str22.array[((signed int )0u)])));
  _ZNSsD1Ev((&llvm_cbe_tmp__7));
  llvm_cbe_tmp__55 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str25.array[((signed int )0u)])));
  llvm_cbe_tmp__56 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__55, ((&_OC_str26.array[((signed int )0u)])));
  llvm_cbe_tmp__8 = _ZNKSs6substrEjj((&llvm_cbe_s1), 15u, 4294967295u);
  llvm_cbe_tmp__57 = _ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(llvm_cbe_tmp__56, (&llvm_cbe_tmp__8));
  llvm_cbe_tmp__58 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(llvm_cbe_tmp__57, ((unsigned char )10));
  _ZNSsD1Ev((&llvm_cbe_tmp__8));
  llvm_cbe_tmp__59 = _Znwj(4u);
  llvm_cbe_tmp__60 = ((struct l_class_OC_std_KD__KD_basic_string *)llvm_cbe_tmp__59);
  _ZNSsC1ERKSs(llvm_cbe_tmp__60, (&llvm_cbe_s1));
  *(&llvm_cbe_s4Ptr) = llvm_cbe_tmp__60;
  llvm_cbe_tmp__61 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str27.array[((signed int )0u)])));
  llvm_cbe_tmp__62 = *(&llvm_cbe_s4Ptr);
  llvm_cbe_tmp__63 = _ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(llvm_cbe_tmp__61, llvm_cbe_tmp__62);
  llvm_cbe_tmp__64 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__63, ((&_OC_str22.array[((signed int )0u)])));
  llvm_cbe_tmp__65 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str28.array[((signed int )0u)])));
  llvm_cbe_tmp__66 = *(&llvm_cbe_s4Ptr);
  llvm_cbe_tmp__67 = *(&llvm_cbe_s4Ptr);
  llvm_cbe_tmp__68 = _ZNSsaSERKSs(llvm_cbe_tmp__66, llvm_cbe_tmp__67);
  llvm_cbe_tmp__69 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str29.array[((signed int )0u)])));
  llvm_cbe_tmp__70 = *(&llvm_cbe_s4Ptr);
  llvm_cbe_tmp__71 = _ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(llvm_cbe_tmp__69, llvm_cbe_tmp__70);
  llvm_cbe_tmp__72 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(llvm_cbe_tmp__71, ((unsigned char )10));
  llvm_cbe_tmp__73 = *(&llvm_cbe_s4Ptr);
  if ((llvm_cbe_tmp__73 == ((struct l_class_OC_std_KD__KD_basic_string *)/*NULL*/0))) {
    goto llvm_cbe_tmp__85;
  } else {
    goto llvm_cbe_tmp__86;
  }

llvm_cbe_tmp__86:
  _ZNSsD1Ev(llvm_cbe_tmp__73);
  _ZdlPv((((unsigned char *)llvm_cbe_tmp__73)));
  goto llvm_cbe_tmp__85;

llvm_cbe_tmp__85:
  llvm_cbe_tmp__74 = _ZNSsixEj((&llvm_cbe_s1), 0u);
  *llvm_cbe_tmp__74 = ((unsigned char )72);
  llvm_cbe_tmp__75 = _ZNSsixEj((&llvm_cbe_s1), 6u);
  *llvm_cbe_tmp__75 = ((unsigned char )66);
  llvm_cbe_tmp__76 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str30.array[((signed int )0u)])));
  llvm_cbe_tmp__77 = _ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKSbIS4_S5_T1_E(llvm_cbe_tmp__76, (&llvm_cbe_s1));
  llvm_cbe_tmp__78 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(llvm_cbe_tmp__77, ((&_OC_str22.array[((signed int )0u)])));
  llvm_cbe_tmp__79 = _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc((&_ZSt4cout), ((&_OC_str31.array[((signed int )0u)])));
  llvm_cbe_tmp__80 = _ZNSolsEPFRSoS_E(llvm_cbe_tmp__79, _ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_);
  llvm_cbe_tmp__81 = _ZNSs2atEj((&llvm_cbe_s1), 30u);
  *llvm_cbe_tmp__81 = ((unsigned char )100);
  *(&llvm_cbe_tmp__2) = 0u;
  *(&llvm_cbe_tmp__9) = 1u;
  _ZNSsD1Ev((&llvm_cbe_s3));
  _ZNSsD1Ev((&llvm_cbe_s2));
  _ZNSsD1Ev((&llvm_cbe_s1));
  llvm_cbe_tmp__82 = *(&llvm_cbe_tmp__2);
  return llvm_cbe_tmp__82;
}


bool _ZSteqIcEN9__gnu_cxx11__enable_ifIXsrSt9__is_charIT_E7__valueEbE6__typeERKSbIS3_St11char_traitsIS3_ESaIS3_EESC_(struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___lhs, struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___rhs) {
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__87;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__88;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__89;
  unsigned int llvm_cbe_tmp__90;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__91;
  unsigned int llvm_cbe_tmp__92;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__93;
  unsigned char *llvm_cbe_tmp__94;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__95;
  unsigned char *llvm_cbe_tmp__96;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__97;
  unsigned int llvm_cbe_tmp__98;
  unsigned int llvm_cbe_tmp__99;

  *(&llvm_cbe_tmp__87) = llvm_cbe___lhs;
  *(&llvm_cbe_tmp__88) = llvm_cbe___rhs;
  llvm_cbe_tmp__89 = *(&llvm_cbe_tmp__87);
  llvm_cbe_tmp__90 = _ZNKSs4sizeEv(llvm_cbe_tmp__89);
  llvm_cbe_tmp__91 = *(&llvm_cbe_tmp__88);
  llvm_cbe_tmp__92 = _ZNKSs4sizeEv(llvm_cbe_tmp__91);
  if ((llvm_cbe_tmp__90 == llvm_cbe_tmp__92)) {
    goto llvm_cbe_tmp__100;
  } else {
    goto llvm_cbe_tmp__101;
  }

llvm_cbe_tmp__100:
  llvm_cbe_tmp__93 = *(&llvm_cbe_tmp__87);
  llvm_cbe_tmp__94 = _ZNKSs4dataEv(llvm_cbe_tmp__93);
  llvm_cbe_tmp__95 = *(&llvm_cbe_tmp__88);
  llvm_cbe_tmp__96 = _ZNKSs4dataEv(llvm_cbe_tmp__95);
  llvm_cbe_tmp__97 = *(&llvm_cbe_tmp__87);
  llvm_cbe_tmp__98 = _ZNKSs4sizeEv(llvm_cbe_tmp__97);
  llvm_cbe_tmp__99 = _ZNSt11char_traitsIcE7compareEPKcS2_j(llvm_cbe_tmp__94, llvm_cbe_tmp__96, llvm_cbe_tmp__98);
  return ((((llvm_cbe_tmp__99 != 0u) ^ 1)&1));
llvm_cbe_tmp__101:
  return 0;
}


bool _ZStneIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_(struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___lhs, struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___rhs) {
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__102;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__103;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__104;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__105;
  bool llvm_cbe_tmp__106;

  *(&llvm_cbe_tmp__102) = llvm_cbe___lhs;
  *(&llvm_cbe_tmp__103) = llvm_cbe___rhs;
  llvm_cbe_tmp__104 = *(&llvm_cbe_tmp__102);
  llvm_cbe_tmp__105 = *(&llvm_cbe_tmp__103);
  llvm_cbe_tmp__106 = ((_ZSteqIcEN9__gnu_cxx11__enable_ifIXsrSt9__is_charIT_E7__valueEbE6__typeERKSbIS3_St11char_traitsIS3_ESaIS3_EESC_(llvm_cbe_tmp__104, llvm_cbe_tmp__105))&1);
  return (((llvm_cbe_tmp__106 ^ 1)&1));
}


bool _ZStgtIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_(struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___lhs, struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___rhs) {
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__107;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__108;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__109;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__110;
  unsigned int llvm_cbe_tmp__111;

  *(&llvm_cbe_tmp__107) = llvm_cbe___lhs;
  *(&llvm_cbe_tmp__108) = llvm_cbe___rhs;
  llvm_cbe_tmp__109 = *(&llvm_cbe_tmp__107);
  llvm_cbe_tmp__110 = *(&llvm_cbe_tmp__108);
  llvm_cbe_tmp__111 = _ZNKSs7compareERKSs(llvm_cbe_tmp__109, llvm_cbe_tmp__110);
  return (((signed int )llvm_cbe_tmp__111) > ((signed int )0u));
}


bool _ZStltIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_(struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___lhs, struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___rhs) {
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__112;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__113;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__114;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__115;
  unsigned int llvm_cbe_tmp__116;

  *(&llvm_cbe_tmp__112) = llvm_cbe___lhs;
  *(&llvm_cbe_tmp__113) = llvm_cbe___rhs;
  llvm_cbe_tmp__114 = *(&llvm_cbe_tmp__112);
  llvm_cbe_tmp__115 = *(&llvm_cbe_tmp__113);
  llvm_cbe_tmp__116 = _ZNKSs7compareERKSs(llvm_cbe_tmp__114, llvm_cbe_tmp__115);
  return (((signed int )llvm_cbe_tmp__116) < ((signed int )0u));
}


bool _ZStgeIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_(struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___lhs, struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___rhs) {
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__117;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__118;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__119;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__120;
  unsigned int llvm_cbe_tmp__121;

  *(&llvm_cbe_tmp__117) = llvm_cbe___lhs;
  *(&llvm_cbe_tmp__118) = llvm_cbe___rhs;
  llvm_cbe_tmp__119 = *(&llvm_cbe_tmp__117);
  llvm_cbe_tmp__120 = *(&llvm_cbe_tmp__118);
  llvm_cbe_tmp__121 = _ZNKSs7compareERKSs(llvm_cbe_tmp__119, llvm_cbe_tmp__120);
  return (((signed int )llvm_cbe_tmp__121) >= ((signed int )0u));
}


bool _ZStleIcSt11char_traitsIcESaIcEEbRKSbIT_T0_T1_ES8_(struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___lhs, struct l_class_OC_std_KD__KD_basic_string *llvm_cbe___rhs) {
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__122;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__123;    /* Address-exposed local */
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__124;
  struct l_class_OC_std_KD__KD_basic_string *llvm_cbe_tmp__125;
  unsigned int llvm_cbe_tmp__126;

  *(&llvm_cbe_tmp__122) = llvm_cbe___lhs;
  *(&llvm_cbe_tmp__123) = llvm_cbe___rhs;
  llvm_cbe_tmp__124 = *(&llvm_cbe_tmp__122);
  llvm_cbe_tmp__125 = *(&llvm_cbe_tmp__123);
  llvm_cbe_tmp__126 = _ZNKSs7compareERKSs(llvm_cbe_tmp__124, llvm_cbe_tmp__125);
  return (((signed int )llvm_cbe_tmp__126) <= ((signed int )0u));
}


unsigned int _ZNSt11char_traitsIcE7compareEPKcS2_j(unsigned char *llvm_cbe___s1, unsigned char *llvm_cbe___s2, unsigned int llvm_cbe___n) {
  unsigned char *llvm_cbe_tmp__127;    /* Address-exposed local */
  unsigned char *llvm_cbe_tmp__128;    /* Address-exposed local */
  unsigned int llvm_cbe_tmp__129;    /* Address-exposed local */
  unsigned char *llvm_cbe_tmp__130;
  unsigned char *llvm_cbe_tmp__131;
  unsigned int llvm_cbe_tmp__132;
  unsigned int llvm_cbe_tmp__133;

  *(&llvm_cbe_tmp__127) = llvm_cbe___s1;
  *(&llvm_cbe_tmp__128) = llvm_cbe___s2;
  *(&llvm_cbe_tmp__129) = llvm_cbe___n;
  llvm_cbe_tmp__130 = *(&llvm_cbe_tmp__127);
  llvm_cbe_tmp__131 = *(&llvm_cbe_tmp__128);
  llvm_cbe_tmp__132 = *(&llvm_cbe_tmp__129);
  llvm_cbe_tmp__133 = memcmp(llvm_cbe_tmp__130, llvm_cbe_tmp__131, llvm_cbe_tmp__132);
  return llvm_cbe_tmp__133;
}


static void _GLOBAL__I_a(void) {
  __cxx_global_var_init();
  return;
}

