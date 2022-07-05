#include <langapi/mode.h>

const mode_table_et mode_table[] = {
  LANGAPI_MODE_CLANG_C,
  LANGAPI_MODE_CLANG_CPP,
// put a new mode before old-frontend,
// otherwise language_uit::parse() will return different mode when old-frontend is enabled
#ifdef ENABLE_SOLIDITY_FRONTEND
  LANGAPI_MODE_SOLAST,
#endif

#ifdef ENABLE_JIMPLE_FRONTEND
  LANGAPI_HAVE_MODE_JIMPLE,
#endif
#ifdef ENABLE_OLD_FRONTEND
  LANGAPI_MODE_C,
  LANGAPI_MODE_CPP,
#endif
  LANGAPI_MODE_END};

extern "C" const uint8_t buildidstring_buf[];
extern "C" const uint8_t *const esbmc_version_string = buildidstring_buf;
