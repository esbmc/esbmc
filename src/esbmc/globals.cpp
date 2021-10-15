#include <langapi/mode.h>

const mode_table_et mode_table[] = {
  LANGAPI_HAVE_MODE_CLANG_C,
  LANGAPI_HAVE_MODE_CLANG_CPP,
  // put a new mode before old-frontend,
  // otherwise language_uit::parse() will return different mode when old-frontend is enabled
  LANGAPI_HAVE_MODE_SOLAST,
#ifdef ENABLE_OLD_FRONTEND
  LANGAPI_HAVE_MODE_C,
  LANGAPI_HAVE_MODE_CPP,
#endif
  LANGAPI_HAVE_MODE_END};

extern "C" uint8_t buildidstring_buf[1];
#ifdef _WIN32
extern "C" uint8_t *esbmc_version_string = buildidstring_buf;
#else
uint8_t *esbmc_version_string = buildidstring_buf;
#endif
