#include <esbmc/globals.h>
#include <langapi/mode.h>

#include <c2goto/cprover_library.h>
#include <clang-c-frontend/clang_c_language.h>
#include <clang-cpp-frontend/esbmc_internal_cpp.h>
#ifdef ENABLE_PYTHON_FRONTEND
#  include <python-frontend/python_language.h>
#endif

void register_bundled_files()
{
  clang_c_languaget::register_bundled();
  esbmct::register_bundled_cpp();
  register_bundled_libc();
#ifdef ENABLE_PYTHON_FRONTEND
  python_languaget::register_bundled();
#endif
}

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
#ifdef ENABLE_PYTHON_FRONTEND
  LANGAPI_MODE_PYTHON,
#endif
#ifdef ENABLE_LD_FRONTEND
  LANGAPI_MODE_LD,
#endif
  LANGAPI_MODE_END};
