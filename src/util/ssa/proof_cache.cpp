#include <util/ssa/proof_cache.h>

#include <ac_config.h>
#include <fmt/format.h>
#include <util/config/config.h>
#include <util/message/message.h>
#include <util/ssa/fingerprint.h>

#include <boost/dll/runtime_symbol_info.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

#ifdef _WIN32
#  include <process.h>
#else
#  include <unistd.h>
#endif

namespace
{
int current_pid()
{
#ifdef _WIN32
  return _getpid();
#else
  return getpid();
#endif
}

/// FNV-1a over a file's bytes, folded a word at a time because the file is the
/// whole ESBMC binary; empty when it could not be read through.
std::optional<uint64_t> file_digest(const std::filesystem::path &path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return std::nullopt;

  uint64_t h = 0xcbf29ce484222325ULL;
  uintmax_t length = 0;
  // A whole number of words, so only the final read can leave a part-word.
  std::vector<char> buf(1 << 16);
  const auto block = static_cast<std::streamsize>(buf.size());
  while (in.read(buf.data(), block) || in.gcount() > 0)
  {
    const size_t n = static_cast<size_t>(in.gcount());
    length += n;

    size_t i = 0;
    for (; i + sizeof(uint64_t) <= n; i += sizeof(uint64_t))
    {
      uint64_t word;
      std::memcpy(&word, buf.data() + i, sizeof(word));
      h = (h ^ word) * 0x100000001b3ULL;
      h ^= h >> 29;
    }
    for (; i < n; ++i)
      h = (h ^ static_cast<unsigned char>(buf[i])) * 0x100000001b3ULL;
  }

  if (in.bad())
    return std::nullopt;
  // Two files whose bytes hash alike still differ if their lengths do.
  return h ^ static_cast<uint64_t>(length);
}

/// scripts/buildidobj.py stamps "(dirty tree)" when the working tree carried
/// uncommitted changes and "no-hash" when git was unavailable. Either way the
/// ID names a class of builds rather than one, so it cannot key a proof on its
/// own.
bool names_one_build(const std::string &build_id)
{
  return build_id.find("(dirty tree)") == std::string::npos &&
         build_id.find("no-hash") == std::string::npos;
}
} // namespace

std::string proof_cache_inactive_reason(const optionst &options)
{
  // A stored proof stands in for solving only where the claim's sliced cone is
  // everything its verdict depends on. A coverage probe asks whether a
  // location is reachable rather than whether a property holds, and its
  // verdict feeds the coverage bookkeeping rather than a report of its own;
  // --ltl reaches one verdict over the whole equation; and --smt-during-symex
  // never builds the per-claim equation a key is taken from.
  if (options.get_bool_option("coverage-measurement"))
    return "coverage measurement";
  if (options.get_bool_option("dead-code-check"))
    return "--dead-code-check";
  if (options.get_bool_option("ltl"))
    return "--ltl";
  if (options.get_bool_option("smt-during-symex"))
    return "--smt-during-symex";
  return "";
}

void report_proof_cache_inactive(const std::string &why)
{
  // multi_property_check is entered per thread interleaving and per k step, so
  // repeating the line would bury the report it explains.
  static std::atomic<bool> reported{false};
  if (!reported.exchange(true))
    log_warning(
      "Proof cache: inactive ({}); every claim will be solved and none stored",
      why);
}

bool proof_cache_contradicted(bool was_hit, bool counterexample_found)
{
  return was_hit && counterexample_found;
}

std::string proof_cache_build_identity(const std::string &build_id)
{
  if (names_one_build(build_id))
    return build_id;

  // Only the binary itself distinguishes two builds of the same commit from a
  // tree git cannot name -- the case a developer of ESBMC is in all day. It is
  // one read of the executable per run, paid only here.
  boost::system::error_code ec;
  const boost::filesystem::path self = boost::dll::program_location(ec);
  if (ec)
    return "";

  const std::optional<uint64_t> digest = file_digest(self.string());
  if (!digest)
    return "";

  return fmt::format("{} exe {:016x}", build_id, *digest);
}

std::string
proof_cache_context(const optionst &options, const std::string &build_identity)
{
  std::ostringstream ctx;
  ctx << "esbmc " << ESBMC_VERSION << '\n';
  ctx << "build " << build_identity << '\n';
  // Which solver a run uses is only in the option set when it was named on the
  // command line; pick_default_solver() takes the rest from what this build
  // enabled, and never writes it back. Two builds of one commit that enabled
  // different backends would otherwise share every key.
  ctx << "solvers " << ESBMC_AVAILABLE_SOLVERS << '\n';

  // Every option is folded in unabridged except the cache's own controls, the
  // source path, and options that reach nothing but the report -- the path is
  // stripped from symbol names before a cone is digested. Curating a wider
  // exclusion list is how a cache like this goes unsound, so an addition here
  // needs its use sites read: `color` and `ascii-report` are auto-detected
  // from the terminal and the locale, so leaving them in made an interactive
  // run and a piped one disagree on every key.
  static const std::set<std::string> not_semantic = {
    "proof-cache",
    "proof-cache-verify",
    "claim-fingerprint-dump",
    "input-file",
    "ascii-report",
    "cex-output",
    "color",
    "file-output",
    "log-message",
    "quiet",
    "verbosity"};
  for (const auto &[name, value] : options.option_map)
  {
    if (not_semantic.count(name))
      continue;
    ctx << "opt " << name << '=' << value << '\n';

    // set_option overwrites, so a repeatable option leaves only its last value
    // in option_map and `-DA -DB` would key the same as `-DB`. The earlier
    // values are folded in on top rather than in place: an option the run
    // rewrites afterwards -- k-induction's per-step `unwind` -- must still key
    // on what it currently holds.
    const auto given = options.option_values.find(name);
    if (given != options.option_values.end() && given->second.size() > 1)
      for (const std::string &v : given->second)
        ctx << "optv " << name << '=' << v << '\n';
  }

  // The data model is not in the option set: absent a --32 or --i386-linux
  // style flag, configt::this_architecture() and this_operating_system() take
  // it from the machine ESBMC was built on. Two hosts sharing one cache
  // directory -- what the docs suggest for CI -- would otherwise agree on
  // every key while disagreeing on type widths, libc models and predefined
  // macros.
  const auto &c = config.ansi_c;
  ctx << "lang " << static_cast<int>(config.language.lid) << ' '
      << config.language.std << ' ' << static_cast<int>(config.language.c_std)
      << ' ' << static_cast<int>(config.language.cpp_std) << '\n';
  ctx << "target " << c.target.to_string() << '\n';
  ctx << "model " << c.word_size << ' ' << c.bool_width << ' ' << c.char_width
      << ' ' << c.short_int_width << ' ' << c.int_width << ' '
      << c.long_int_width << ' ' << c.long_long_int_width << ' '
      << c.int_128_width << ' ' << c.address_width << ' ' << c.pointer_width()
      << ' ' << c.pointer_diff_width << ' ' << c.single_width << ' '
      << c.double_width << ' ' << c.long_double_width << ' ' << c.wchar_t_width
      << ' ' << c.char_is_unsigned << ' ' << static_cast<int>(c.cheri) << ' '
      << c.cheri_concentrate << ' ' << static_cast<int>(c.endianess) << ' '
      << static_cast<int>(c.lib) << ' ' << c.locale_name << '\n';

  return ctx.str();
}

proof_cachet::proof_cachet(
  const std::string &dir,
  const optionst &options,
  const std::string &build_identity)
  : dir(dir),
    context_hash(fingerprint_hash(proof_cache_context(options, build_identity)))
{
  std::error_code ec;
  std::filesystem::create_directories(this->dir, ec);
  if (ec)
    log_warning("Proof cache: cannot create {}: {}", dir, ec.message());
}

std::filesystem::path
proof_cachet::entry_path(const std::string &cone_key) const
{
  const std::string key = fmt::format("{:016x}{}", context_hash, cone_key);
  return dir / key.substr(0, 2) / (key + ".proof");
}

bool proof_cachet::proved(const std::string &cone_key) const
{
  std::error_code ec;
  if (std::filesystem::exists(entry_path(cone_key), ec))
  {
    ++hit_count;
    return true;
  }

  ++miss_count;
  return false;
}

void proof_cachet::record(const std::string &cone_key) const
{
  const std::filesystem::path path = entry_path(cone_key);

  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);

  // Written aside and renamed so a concurrent reader never sees a half-created
  // entry.
  const std::filesystem::path tmp = fmt::format(
    "{}.tmp{}.{:x}",
    path.string(),
    current_pid(),
    std::hash<std::thread::id>{}(std::this_thread::get_id()));
  std::ofstream out(tmp, std::ios::binary);
  if (!out)
    return;
  out.close();

  std::filesystem::rename(tmp, path, ec);
  if (ec)
    std::filesystem::remove(tmp, ec);
}
