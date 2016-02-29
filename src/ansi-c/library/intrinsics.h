void __ESBMC_assert(_Bool x, char *msg);
void __ESBMC_assume(_Bool x);
void __ESBMC_atomic_begin(void);
void __ESBMC_atomic_end(void);
void __ESBMC_yield(void);
void __ESBMC_switch_to(unsigned int tid);

int __ESBMC_abs(int x);
long int __ESBMC_labs(long int x);
double __ESBMC_fabs(double x);
float __ESBMC_fabsf(float x);
long double __ESBMC_fabsl(long double x);

_Bool __ESBMC_isfinite(double x);
_Bool __ESBMC_isinf(double x);
_Bool __ESBMC_isnan(double x);
_Bool __ESBMC_isnormal(double x);
_Bool __ESBMC_sign(double x);
extern int __ESBMC_rounding_mode;

typedef void *(*__ESBMC_thread_start_func_type)(void *);


void __ESBMC_terminate_thread(void);
