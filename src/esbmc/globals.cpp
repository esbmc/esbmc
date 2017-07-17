#include <esbmc/version.h>
#include <langapi/mode.h>

const mode_table_et mode_table[] =
{
#ifndef WITHOUT_CLANG
  LANGAPI_HAVE_MODE_CLANG_C,
#endif
  LANGAPI_HAVE_MODE_C,
#ifndef WITHOUT_CLANG
  LANGAPI_HAVE_MODE_CPP,
#endif
  LANGAPI_HAVE_MODE_CPP,
  LANGAPI_HAVE_MODE_END
};

extern "C" uint8_t buildidstring_buf[1];
uint8_t *esbmc_version_string = buildidstring_buf;

extern "C" {
uint64_t esbmc_version = ESBMC_VERSION_CONST;
}
