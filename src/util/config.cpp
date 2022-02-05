/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <algorithm>
#include <regex>

#include <util/config.h>

configt config;

void configt::ansi_ct::set_data_model(enum data_model dm)
{
  auto next = [m = static_cast<uint64_t>(dm)]() mutable {
    unsigned r = m & 0xff;
    m >>= 8;
    return r;
  };

  char_width = next();
  short_int_width = next();
  int_width = next();
  long_int_width = next();
  pointer_width = next();
  word_size = next();
  long_double_width = next();

  long_long_int_width = 64;
  bool_width = char_width;
  pointer_diff_width = pointer_width;
  single_width = 32;
  double_width = 64;
  char_is_unsigned = false;
}

namespace
{
struct eregex : std::regex
{
  eregex(std::string pat)
    : std::regex(std::move(pat), std::regex_constants::extended)
  {
  }
};
} // namespace

static const eregex WINDOWS_ABI("mingw.*|win[0-9]{2}|windows|msys");
static const eregex FREEBSD("k?freebsd.*");
static const eregex MACOS("macos|osx.*");
static const eregex X86("i[3456]86|x86_64|x64");
static const eregex ARM("(arm|thumb|aarch64c?)(eb|_be)?");
static const eregex MIPS("mips(64|isa64|isa64sb1)?(r[0-9]+)?(el|le)?.*");
static const eregex POWERPC("(ppc|powerpc)(64)?(le)?");

static configt::ansi_ct::endianesst
arch_endianness(const std::string &arch, const messaget &msg)
{
  if(std::regex_match(arch, X86) || arch == "riscv32" || arch == "riscv64")
    return configt::ansi_ct::IS_LITTLE_ENDIAN;
  std::smatch r;
  if(std::regex_match(arch, r, ARM))
    return r.length(2) > 0 ? configt::ansi_ct::IS_BIG_ENDIAN
                           : configt::ansi_ct::IS_LITTLE_ENDIAN;
  if(std::regex_match(arch, r, MIPS))
    return r.length(3) > 0 ? configt::ansi_ct::IS_LITTLE_ENDIAN
                           : configt::ansi_ct::IS_BIG_ENDIAN;
  if(std::regex_match(arch, r, POWERPC))
    return r.length(3) > 0 ? configt::ansi_ct::IS_LITTLE_ENDIAN
                           : configt::ansi_ct::IS_BIG_ENDIAN;
  if(arch == "none")
    return configt::ansi_ct::NO_ENDIANESS;
  msg.error("unknown arch '" + arch + "', cannot determine endianness\n");
  abort();
}

bool configt::triple::is_windows_abi() const
{
  return std::regex_match(os, WINDOWS_ABI);
}

bool configt::triple::is_freebsd() const
{
  return std::regex_match(os, FREEBSD);
}

bool configt::triple::is_macos() const
{
  return std::regex_match(os, MACOS);
}

std::string configt::triple::to_string() const
{
  return arch + "-" + vendor + "-" + os + (flavor.empty() ? "" : "-" + flavor);
}

bool configt::set(const cmdlinet &cmdline, const messaget &msg)
{
  if(cmdline.isset("function"))
    main = cmdline.getval("function");

  if(cmdline.isset("define"))
    ansi_c.defines = cmdline.get_values("define");

  if(cmdline.isset("include"))
    ansi_c.include_paths = cmdline.get_values("include");

  if(cmdline.isset("idirafter"))
    ansi_c.idirafter_paths = cmdline.get_values("idirafter");

  if(cmdline.isset("force"))
    ansi_c.forces = cmdline.get_values("force");

  if(cmdline.isset("warning"))
    ansi_c.warnings = cmdline.get_values("warning");

  if(cmdline.isset("floatbv") && cmdline.isset("fixedbv"))
  {
    msg.error("Can't set both floatbv and fixedbv modes");
    return true;
  }

  ansi_c.use_fixed_for_float = cmdline.isset("fixedbv");

  // this is the default
  std::string arch = this_architecture(), os = this_operating_system();
  int req_target = 0;

  if(cmdline.isset("i386-linux"))
  {
    arch = "i386";
    os = "linux";
    req_target++;
  }

  if(cmdline.isset("i386-win32"))
  {
    arch = "i386";
    os = "win32";
    req_target++;
  }

  if(cmdline.isset("i386-macos"))
  {
    arch = "i386";
    os = "macos";
    req_target++;
  }

  if(cmdline.isset("ppc-macos"))
  {
    arch = "ppc";
    os = "macos";
    req_target++;
  }

  if(cmdline.isset("no-arch"))
  {
    arch = "none";
    os = "elf";
    req_target++;
  }

  if(req_target > 1)
  {
    msg.error(
      "only at most one target can be specified via "
      "--i386-{win32,macos,linux}, --ppc-macos and --no-arch\n");
    return true;
  }

  ansi_c.target.arch = arch;
  ansi_c.target.os = os;

  bool have_16 = cmdline.isset("16");
  bool have_32 = cmdline.isset("32");
  bool have_64 = cmdline.isset("64");

  if(have_16 + have_32 + have_64 > 1)
  {
    msg.error("Only one of --16, --32 and --64 is supported");
    return true;
  }

  enum data_model dm;
  if(have_16)
    dm = LP32;
  else if(have_32)
    dm = ILP32;
  else
    dm = ansi_c.target.is_windows_abi() ? LLP64 : LP64;
  ansi_c.set_data_model(dm);

  if(cmdline.isset("little-endian") && cmdline.isset("big-endian"))
  {
    msg.error("Can't set both little and big endian modes");
    return true;
  }

  ansi_c.endianess = cmdline.isset("little-endian") ? ansi_ct::IS_LITTLE_ENDIAN
                     : cmdline.isset("big-endian")
                       ? ansi_ct::IS_BIG_ENDIAN
                       : arch_endianness(ansi_c.target.arch, msg);

  ansi_c.lib = ansi_c.target.arch == "none" || cmdline.isset("no-library")
                 ? ansi_ct::LIB_NONE
                 : ansi_ct::LIB_FULL;

  /* wchar_t is ABI-dependent: Windows: 16, other: 32 */
  ansi_c.wchar_t_width = ansi_c.target.is_windows_abi() ? 16 : 32;

  ansi_c.char_is_unsigned =
    (std::find(
       ansi_c.forces.begin(),
       ansi_c.forces.end(),
       std::string("unsigned-char")) != ansi_c.forces.end());

  return false;
}

std::string configt::this_architecture()
{
  std::string this_arch;

  /* We're passing this down to clang via -target */

  /* References:
   * http://wiki.debian.org/ArchitectureSpecificsMemo
   * https://sourceforge.net/p/predef/wiki/Architectures/
   */

#ifdef __alpha__
  this_arch = "alpha";
#elif __thumb__
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  this_arch = "thumbeb"
#else
  this_arch = "thumb";
#endif
#elif __aarch64__
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  this_arch = "aarch64_be"
#else
  this_arch = "aarch64";
#endif
#elif __arm__
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  this_arch = "armeb";
#else
  this_arch = "arm";
#endif
#elif __mipsel__
#if _MIPS_SIM == _ABIO32
  this_arch = "mipsel";
#elif _MIPS_SIM == _ABIN32
  this_arch = "mipsn32el";
#else
  this_arch = "mips64el";
#endif
#elif __mips__
#if _MIPS_SIM == _ABIO32
  this_arch = "mips";
#elif _MIPS_SIM == _ABIN32
  this_arch = "mipsn32";
#else
  this_arch = "mips64";
#endif
#elif __powerpc__
#if defined(__ppc64__) || defined(__PPC64__) || defined(__powerpc64__) ||      \
  defined(__POWERPC64__)
#ifdef __LITTLE_ENDIAN__
  this_arch = "ppc64le";
#else
  this_arch = "ppc64";
#endif
#else
  this_arch = "powerpc";
#endif
#elif __sparc__
#ifdef __arch64__
  this_arch = "sparc64";
#else
  this_arch = "sparc";
#endif
#elif __ia64__
  this_arch = "ia64";
#elif __s390x__
  this_arch = "s390x";
#elif __s390__
  this_arch = "s390";
#elif __x86_64__
#ifdef __ILP32__
  this_arch = "x32"; // variant of x86_64 with 32-bit pointers
#else
  this_arch = "x86_64";
#endif
#elif __i386__
  this_arch = "i386";
#elif _WIN64
  this_arch = "x86_64";
#elif _WIN32
  this_arch = "i386";
#else
  // something new and unknown!
  this_arch = "unknown";
#endif

  return this_arch;
}

std::string configt::this_operating_system()
{
  std::string this_os;

#ifdef _WIN32
  this_os = "windows";
#elif __APPLE__
  this_os = "macos";
#elif __FreeBSD__
  this_os = "freebsd";
#elif __linux__
  this_os = "linux";
#elif __SVR4
  this_os = "solaris";
#else
  this_os = "unknown";
#endif

  return this_os;
}

configt::triple configt::host()
{
  return {this_architecture(), "unknown", this_operating_system(), ""};
}
