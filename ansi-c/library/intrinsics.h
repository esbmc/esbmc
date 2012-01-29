void __ESBMC_assert(_Bool x, char *msg);
void __ESBMC_assume(_Bool x);
void __ESBMC_atomic_begin(void);
void __ESBMC_atomic_end(void);
void __ESBMC_yield(void);
void __ESBMC_switch_to(unsigned int tid);

int __ESBMC_buffer_size(void *b);
_Bool __ESBMC_is_zero_string(char *s);
int __ESBMC_zero_string_length(char *s);

int __ESBMC_abs(int x);
long int __ESBMC_labs(long int x);
double __ESBMC_fabs(double x);
float __ESBMC_fabsf(float x);
long double __ESBMC_fabsl(long double x);

int __ESBMC_isfinite(double x);
int __ESBMC_isinf(double x);
int __ESBMC_isnan(double x);
int __ESBMC_isnormal(double x);
int __ESBMC_sign(double x);
extern int __ESBMC_rounding_mode;

typedef void *(*__ESBMC_thread_start_func_type)(void *);
void *__ESBMC_get_thread_start_arg();
__ESBMC_thread_start_func_type __ESBMC_get_thread_start_func();
void __ESBMC_set_next_thread_start_arg(void *);
void __ESBMC_set_next_thread_start_func(__ESBMC_thread_start_func_type);
void __ESBMC_spawn_thread(void (*)(void));
