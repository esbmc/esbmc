#define PARI_TUNE

#ifdef PARI_TUNE
extern long AGM_ATAN_LIMIT;
extern long DIVRR_GMP_LIMIT;
extern long EXPNEWTON_LIMIT;
extern long EXTGCD_HALFGCD_LIMIT;
extern long F2x_MUL_KARATSUBA_LIMIT;
extern long F2x_MUL_MULII_LIMIT;
extern long F2xqX_BARRETT_LIMIT;
extern long F2xqX_DIVREM_BARRETT_LIMIT;
extern long F2xqX_EXTGCD_LIMIT;
extern long F2xqX_GCD_LIMIT;
extern long F2xqX_HALFGCD_LIMIT;
extern long F2xqX_INVBARRETT_LIMIT;
extern long F2xqX_REM_BARRETT_LIMIT;
extern long Flx_BARRETT2_LIMIT;
extern long Flx_BARRETT_LIMIT;
extern long Flx_DIVREM2_BARRETT_LIMIT;
extern long Flx_DIVREM_BARRETT_LIMIT;
extern long Flx_EXTGCD2_LIMIT;
extern long Flx_EXTGCD_LIMIT;
extern long Flx_GCD2_LIMIT;
extern long Flx_GCD_LIMIT;
extern long Flx_HALFGCD2_LIMIT;
extern long Flx_HALFGCD_LIMIT;
extern long Flx_INVBARRETT2_LIMIT;
extern long Flx_INVBARRETT_LIMIT;
extern long Flx_MUL2_KARATSUBA_LIMIT;
extern long Flx_MUL2_MULII_LIMIT;
extern long Flx_MUL_KARATSUBA_LIMIT;
extern long Flx_MUL_MULII_LIMIT;
extern long Flx_REM2_BARRETT_LIMIT;
extern long Flx_REM_BARRETT_LIMIT;
extern long Flx_SQR2_KARATSUBA_LIMIT;
extern long Flx_SQR2_SQRI_LIMIT;
extern long Flx_SQR_KARATSUBA_LIMIT;
extern long Flx_SQR_SQRI_LIMIT;
extern long FlxqX_BARRETT_LIMIT;
extern long FlxqX_DIVREM_BARRETT_LIMIT;
extern long FlxqX_EXTGCD_LIMIT;
extern long FlxqX_GCD_LIMIT;
extern long FlxqX_HALFGCD_LIMIT;
extern long FlxqX_INVBARRETT_LIMIT;
extern long FlxqX_REM_BARRETT_LIMIT;
extern long FpXQX_BARRETT_LIMIT;
extern long FpXQX_DIVREM_BARRETT_LIMIT;
extern long FpXQX_EXTGCD_LIMIT;
extern long FpXQX_GCD_LIMIT;
extern long FpXQX_HALFGCD_LIMIT;
extern long FpXQX_INVBARRETT_LIMIT;
extern long FpXQX_REM_BARRETT_LIMIT;
extern long FpX_BARRETT_LIMIT;
extern long FpX_DIVREM_BARRETT_LIMIT;
extern long FpX_EXTGCD_LIMIT;
extern long FpX_GCD_LIMIT;
extern long FpX_HALFGCD_LIMIT;
extern long FpX_INVBARRETT_LIMIT;
extern long FpX_REM_BARRETT_LIMIT;
extern long Fp_POW_BARRETT_LIMIT;
extern long Fp_POW_REDC_LIMIT;
extern long GCD_HALFGCD_LIMIT;
extern long HALFGCD_LIMIT;
extern long INVMOD_GMP_LIMIT;
extern long INVNEWTON_LIMIT;
extern long LOGAGMCX_LIMIT;
extern long LOGAGM_LIMIT;
extern long MULII_FFT_LIMIT;
extern long MULII_KARATSUBA_LIMIT;
extern long MULRR_MULII_LIMIT;
extern long RgX_MUL_LIMIT;
extern long RgX_SQR_LIMIT;
extern long SQRI_FFT_LIMIT;
extern long SQRI_KARATSUBA_LIMIT;
extern long SQRR_SQRI_LIMIT;
#else
#  define AGM_ATAN_LIMIT                 __AGM_ATAN_LIMIT
#  define DIVRR_GMP_LIMIT                __DIVRR_GMP_LIMIT
#  define EXPNEWTON_LIMIT                __EXPNEWTON_LIMIT
#  define EXTGCD_HALFGCD_LIMIT           __EXTGCD_HALFGCD_LIMIT
#  define F2x_MUL_KARATSUBA_LIMIT        __F2x_MUL_KARATSUBA_LIMIT
#  define F2x_MUL_MULII_LIMIT            __F2x_MUL_MULII_LIMIT
#  define F2xqX_BARRETT_LIMIT            __F2xqX_BARRETT_LIMIT
#  define F2xqX_DIVREM_BARRETT_LIMIT     __F2xqX_DIVREM_BARRETT_LIMIT
#  define F2xqX_EXTGCD_LIMIT             __F2xqX_EXTGCD_LIMIT
#  define F2xqX_GCD_LIMIT                __F2xqX_GCD_LIMIT
#  define F2xqX_HALFGCD_LIMIT            __F2xqX_HALFGCD_LIMIT
#  define F2xqX_INVBARRETT_LIMIT         __F2xqX_INVBARRETT_LIMIT
#  define F2xqX_REM_BARRETT_LIMIT        __F2xqX_REM_BARRETT_LIMIT
#  define Flx_BARRETT2_LIMIT             __Flx_BARRETT2_LIMIT
#  define Flx_BARRETT_LIMIT              __Flx_BARRETT_LIMIT
#  define Flx_DIVREM2_BARRETT_LIMIT      __Flx_DIVREM2_BARRETT_LIMIT
#  define Flx_DIVREM_BARRETT_LIMIT       __Flx_DIVREM_BARRETT_LIMIT
#  define Flx_EXTGCD2_LIMIT              __Flx_EXTGCD2_LIMIT
#  define Flx_EXTGCD_LIMIT               __Flx_EXTGCD_LIMIT
#  define Flx_GCD2_LIMIT                 __Flx_GCD2_LIMIT
#  define Flx_GCD_LIMIT                  __Flx_GCD_LIMIT
#  define Flx_HALFGCD2_LIMIT             __Flx_HALFGCD2_LIMIT
#  define Flx_HALFGCD_LIMIT              __Flx_HALFGCD_LIMIT
#  define Flx_INVBARRETT2_LIMIT          __Flx_INVBARRETT2_LIMIT
#  define Flx_INVBARRETT_LIMIT           __Flx_INVBARRETT_LIMIT
#  define Flx_MUL2_KARATSUBA_LIMIT       __Flx_MUL2_KARATSUBA_LIMIT
#  define Flx_MUL2_MULII_LIMIT           __Flx_MUL2_MULII_LIMIT
#  define Flx_MUL_KARATSUBA_LIMIT        __Flx_MUL_KARATSUBA_LIMIT
#  define Flx_MUL_MULII_LIMIT            __Flx_MUL_MULII_LIMIT
#  define Flx_REM2_BARRETT_LIMIT         __Flx_REM2_BARRETT_LIMIT
#  define Flx_REM_BARRETT_LIMIT          __Flx_REM_BARRETT_LIMIT
#  define Flx_SQR2_KARATSUBA_LIMIT       __Flx_SQR2_KARATSUBA_LIMIT
#  define Flx_SQR2_SQRI_LIMIT            __Flx_SQR2_SQRI_LIMIT
#  define Flx_SQR_KARATSUBA_LIMIT        __Flx_SQR_KARATSUBA_LIMIT
#  define Flx_SQR_SQRI_LIMIT             __Flx_SQR_SQRI_LIMIT
#  define FlxqX_BARRETT_LIMIT            __FlxqX_BARRETT_LIMIT
#  define FlxqX_DIVREM_BARRETT_LIMIT     __FlxqX_DIVREM_BARRETT_LIMIT
#  define FlxqX_EXTGCD_LIMIT             __FlxqX_EXTGCD_LIMIT
#  define FlxqX_GCD_LIMIT                __FlxqX_GCD_LIMIT
#  define FlxqX_HALFGCD_LIMIT            __FlxqX_HALFGCD_LIMIT
#  define FlxqX_INVBARRETT_LIMIT         __FlxqX_INVBARRETT_LIMIT
#  define FlxqX_REM_BARRETT_LIMIT        __FlxqX_REM_BARRETT_LIMIT
#  define FpXQX_BARRETT_LIMIT            __FpXQX_BARRETT_LIMIT
#  define FpXQX_DIVREM_BARRETT_LIMIT     __FpXQX_DIVREM_BARRETT_LIMIT
#  define FpXQX_EXTGCD_LIMIT             __FpXQX_EXTGCD_LIMIT
#  define FpXQX_GCD_LIMIT                __FpXQX_GCD_LIMIT
#  define FpXQX_HALFGCD_LIMIT            __FpXQX_HALFGCD_LIMIT
#  define FpXQX_INVBARRETT_LIMIT         __FpXQX_INVBARRETT_LIMIT
#  define FpXQX_REM_BARRETT_LIMIT        __FpXQX_REM_BARRETT_LIMIT
#  define FpX_BARRETT_LIMIT              __FpX_BARRETT_LIMIT
#  define FpX_DIVREM_BARRETT_LIMIT       __FpX_DIVREM_BARRETT_LIMIT
#  define FpX_EXTGCD_LIMIT               __FpX_EXTGCD_LIMIT
#  define FpX_GCD_LIMIT                  __FpX_GCD_LIMIT
#  define FpX_HALFGCD_LIMIT              __FpX_HALFGCD_LIMIT
#  define FpX_INVBARRETT_LIMIT           __FpX_INVBARRETT_LIMIT
#  define FpX_REM_BARRETT_LIMIT          __FpX_REM_BARRETT_LIMIT
#  define Fp_POW_BARRETT_LIMIT           __Fp_POW_BARRETT_LIMIT
#  define Fp_POW_REDC_LIMIT              __Fp_POW_REDC_LIMIT
#  define GCD_HALFGCD_LIMIT              __GCD_HALFGCD_LIMIT
#  define HALFGCD_LIMIT                  __HALFGCD_LIMIT
#  define INVMOD_GMP_LIMIT               __INVMOD_GMP_LIMIT
#  define INVNEWTON_LIMIT                __INVNEWTON_LIMIT
#  define LOGAGMCX_LIMIT                 __LOGAGMCX_LIMIT
#  define LOGAGM_LIMIT                   __LOGAGM_LIMIT
#  define MULII_FFT_LIMIT                __MULII_FFT_LIMIT
#  define MULII_KARATSUBA_LIMIT          __MULII_KARATSUBA_LIMIT
#  define MULRR_MULII_LIMIT              __MULRR_MULII_LIMIT
#  define RgX_MUL_LIMIT                  __RgX_MUL_LIMIT
#  define RgX_SQR_LIMIT                  __RgX_SQR_LIMIT
#  define SQRI_FFT_LIMIT                 __SQRI_FFT_LIMIT
#  define SQRI_KARATSUBA_LIMIT           __SQRI_KARATSUBA_LIMIT
#  define SQRR_SQRI_LIMIT                __SQRR_SQRI_LIMIT
#endif
