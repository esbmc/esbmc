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
  LANGAPI_MODE_JIMPLE,
#endif
#ifdef ENABLE_OLD_FRONTEND
  LANGAPI_MODE_C,
  LANGAPI_MODE_CPP,
#endif
#ifdef ENABLE_PYTHON_FRONTEND
  LANGAPI_MODE_PYTHON,
#endif
  LANGAPI_MODE_END};
