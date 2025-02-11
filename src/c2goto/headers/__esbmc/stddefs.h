#pragma once

#ifdef __cplusplus
# define __ESBMC_C_CPP_BEGIN	extern "C" {
# define __ESBMC_C_CPP_END	}
# define __ESBMC_restrict
#else
# define __ESBMC_C_CPP_BEGIN
# define __ESBMC_C_CPP_END
# define __ESBMC_restrict	restrict
#endif

