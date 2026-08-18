#include <util/ssa/vcc_cache.h>

#include <ac_config.h>
#include <fmt/format.h>
#include <util/config/config.h>
#include <util/message/message.h>
#include <util/ssa/fingerprint.h>

#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

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
} // namespace

std::string vcc_cache_context(const optionst &options)
{
  std::ostringstream ctx;
  ctx << "esbmc " << ESBMC_VERSION << '\n';

  // Every option is folded in unabridged except the cache's own controls and
  // the source path, none of which change what is verified -- the path is
  // stripped from symbol names before a cone is digested. Curating a wider
  // exclusion list is how a cache like this goes unsound.
  static const std::set<std::string> not_semantic = {
    "vcc-cache", "vcc-cache-verify", "vcc-fingerprint-dump", "input-file"};
  for (const auto &[name, value] : options.option_map)
    if (!not_semantic.count(name))
      ctx << "opt " << name << '=' << value << '\n';

  const auto &c = config.ansi_c;
  ctx << "lang " << static_cast<int>(config.language.lid) << '\n';
  ctx << "model " << c.word_size << ' ' << c.int_width << ' '
      << c.long_int_width << ' ' << c.short_int_width << ' '
      << c.long_long_int_width << ' ' << c.char_is_unsigned << ' '
      << static_cast<int>(c.endianess) << '\n';

  return ctx.str();
}

vcc_cachet::vcc_cachet(const std::string &dir, const optionst &options)
  : dir(dir), context(vcc_cache_context(options))
{
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec)
    log_warning("VCC cache: cannot create {}: {}", dir, ec.message());
}

std::string vcc_cachet::entry_path(const std::string &cone_text) const
{
  const std::string key =
    fmt::format("{:016x}", fingerprint_hash(context + cone_text));
  return dir + "/" + key.substr(0, 2) + "/" + key + ".vcc";
}

bool vcc_cachet::proved(const std::string &cone_text) const
{
  std::ifstream in(entry_path(cone_text), std::ios::binary);
  if (in)
  {
    std::ostringstream stored;
    stored << in.rdbuf();
    // The stored text is compared in full, so a digest collision is a miss.
    if (stored.str() == cone_text)
    {
      ++hit_count;
      return true;
    }
    log_warning("VCC cache: digest collision, re-solving");
  }

  ++miss_count;
  return false;
}

void vcc_cachet::record(const std::string &cone_text) const
{
  const std::string path = entry_path(cone_text);

  std::error_code ec;
  std::filesystem::create_directories(
    std::filesystem::path(path).parent_path(), ec);

  // Written aside and renamed so a concurrent reader never sees a partial
  // entry and reads it back as a mismatch.
  const std::string tmp = path + ".tmp" + std::to_string(current_pid());
  std::ofstream out(tmp, std::ios::binary);
  if (!out)
    return;
  out << cone_text;
  out.close();

  // A short write would publish an entry that can never match, wasting its
  // slot for good; drop it rather than rename it into place.
  if (!out)
  {
    std::filesystem::remove(tmp, ec);
    return;
  }

  std::filesystem::rename(tmp, path, ec);
  if (ec)
    std::filesystem::remove(tmp, ec);
}
