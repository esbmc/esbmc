#include <c2goto/cprover_library.h>
#include <util/config/config.h>
#include <util/lang/language.h>
#include <clang-c-frontend/AST/vfs_paths.h>
#include <util/base/filesystem.h>
#include <util/message/message.h>

extern "C"
{
#include <libc.h>
#include <libm.h>
#include <libmusl.h>
#include <libmuslhdr.h>
#include <libcpp.h>
#include <headers/libc_hdr.h>
#ifdef ENABLE_PYTHON_FRONTEND
#  include <libpython.h>
#  include <libpythonhdr.h>
#endif
#ifdef ENABLE_SOLIDITY_FRONTEND
#  include <libsolidity.h>
#  include <libsolidityhdr.h>
#endif
#undef ESBMC_FLAIL
}

static const std::string vfs_prefix = clang_vfs_root() + "/libc";
static const std::string vfs_headers = vfs_prefix + "/headers";
static const std::string vfs_library = vfs_prefix + "/library";
static const std::string vfs_libm = vfs_library + "/libm";
static const std::string vfs_musl = vfs_libm + "/musl";
static const std::string vfs_cpp = vfs_library + "/cpp";
static const std::string vfs_python = vfs_library + "/python";
static const std::string vfs_solidity = vfs_library + "/solidity";

/* The build system generates one ESBMC_FLAIL(body, size, name) invocation per
 * bundled file; see scripts/flail.py --macro. */
void register_bundled_libc()
{
  static bool done = false;
  if (done)
    return;
  done = true;

#define ESBMC_FLAIL(body, size, ...)                                           \
  file_operations::filesystemt::get().add_bundled(                             \
    vfs_headers + "/" #__VA_ARGS__, body, size);
#include <headers/libc_hdr.h>
#undef ESBMC_FLAIL

#define ESBMC_FLAIL(body, size, ...)                                           \
  file_operations::filesystemt::get().add_bundled(                             \
    vfs_library + "/" #__VA_ARGS__, body, size);
#include <libc.h>
#undef ESBMC_FLAIL

#define ESBMC_FLAIL(body, size, ...)                                           \
  file_operations::filesystemt::get().add_bundled(                             \
    vfs_libm + "/" #__VA_ARGS__, body, size);
#include <libm.h>
#undef ESBMC_FLAIL

#define ESBMC_FLAIL(body, size, ...)                                           \
  file_operations::filesystemt::get().add_bundled(                             \
    vfs_musl + "/" #__VA_ARGS__, body, size);
#include <libmusl.h>
#undef ESBMC_FLAIL

#define ESBMC_FLAIL(body, size, ...)                                           \
  file_operations::filesystemt::get().add_bundled(                             \
    vfs_musl + "/" #__VA_ARGS__, body, size);
#include <libmuslhdr.h>
#undef ESBMC_FLAIL

#define ESBMC_FLAIL(body, size, ...)                                           \
  file_operations::filesystemt::get().add_bundled(                             \
    vfs_cpp + "/" #__VA_ARGS__, body, size);
#include <libcpp.h>
#undef ESBMC_FLAIL

#ifdef ENABLE_PYTHON_FRONTEND
#  define ESBMC_FLAIL(body, size, ...)                                         \
    file_operations::filesystemt::get().add_bundled(                           \
      vfs_python + "/" #__VA_ARGS__, body, size);
#  include <libpython.h>
#  include <libpythonhdr.h>
#  undef ESBMC_FLAIL
#endif

#ifdef ENABLE_SOLIDITY_FRONTEND
#  define ESBMC_FLAIL(body, size, ...)                                         \
    file_operations::filesystemt::get().add_bundled(                           \
      vfs_solidity + "/" #__VA_ARGS__, body, size);
#  include <libsolidity.h>
#  include <libsolidityhdr.h>
#  undef ESBMC_FLAIL
#endif
}

const std::string *internal_libc_header_dir()
{
  if (config.ansi_c.lib == configt::ansi_ct::libt::LIB_NONE)
    return nullptr;

  register_bundled_libc();
  // Inside esbmc_clang_vfs(), never on disk.
  return &vfs_headers;
}

/* copy of clang_c_convertert::get_filename_from_path(); TODO: unify */
static std::string get_filename_from_path(const std::string &path)
{
  if (path.find_last_of('/') != std::string::npos)
    return path.substr(path.find_last_of('/') + 1);

  return path;
}

void add_bundled_library_sources(contextt &context, const languaget &c_language)
{
  if (!internal_libc_header_dir())
    return;

  bool skip_fenv = false;
#ifdef ESBMC_CHERI_CLANG_MORELLO
  /* Morello is hard-float, which has an incompatible <fenv.h> */
  skip_fenv = config.ansi_c.cheri != configt::ansi_ct::CHERI_OFF;
#endif

  // Parsed straight out of .rodata via esbmc_clang_vfs(); nothing hits disk.
  auto parse = [&](const std::string &path) {
    if (skip_fenv && get_filename_from_path(path) == "fenv.c")
      return;

    languaget *l = c_language.new_language();
    log_status("file {}: Parsing", path);
    if (l->parse(path) || l->typecheck(context, path))
    {
      log_error("error processing internal libc source {}", path);
      abort();
    }
    delete l;
  };

  /* Driven by the generated headers, not filesystemt::list(): list() sorts by
   * path, which would interleave libm with libc and change the order in which
   * they are merged into the context. */
#define ESBMC_FLAIL(body, size, ...) parse(vfs_library + "/" #__VA_ARGS__);
#include <libc.h>
#undef ESBMC_FLAIL

#define ESBMC_FLAIL(body, size, ...) parse(vfs_libm + "/" #__VA_ARGS__);
#include <libm.h>
#undef ESBMC_FLAIL
}
