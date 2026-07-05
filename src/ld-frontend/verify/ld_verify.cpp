#include <ld-frontend/verify/ld_verify.h>
#include <sstream>
#include <vector>
#include <cstdlib>

#include <boost/version.hpp>
#include <boost/filesystem.hpp>
#include <boost/dll/runtime_symbol_info.hpp>

// Boost.Process header layout differs between releases (see the same block in
// src/python-frontend/python_language.cpp).  We use it to run the esbmc binary.
#if defined(__APPLE__) || (BOOST_VERSION == 108700)
#  include <boost/process/v1.hpp>
namespace bp = boost::process::v1;
#elif BOOST_VERSION >= 108800
#  include <boost/process/v1/child.hpp>
#  include <boost/process/v1/io.hpp>
#  include <boost/process/v1/search_path.hpp>
namespace bp = boost::process::v1;
#else
#  include <boost/process.hpp>
namespace bp = boost::process;
#endif

namespace fs = boost::filesystem;

static std::string json_escape(const std::string &s)
{
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s)
  {
    switch (c)
    {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20)
      {
        char buf[8];
        snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
        out += buf;
      }
      else
        out += static_cast<char>(c);
      break;
    }
  }
  return out;
}

std::string LdVerifyResult::to_json() const
{
  auto verdict_str = [&]() -> const char * {
    switch (verdict)
    {
    case Verdict::Safe:
      return "SAFE";
    case Verdict::Violation:
      return "VIOLATION";
    case Verdict::Unknown:
      return "UNKNOWN";
    case Verdict::Incomplete:
      return "INCOMPLETE";
    case Verdict::Error:
      return "ERROR";
    }
    return "UNKNOWN";
  };

  std::ostringstream s;
  s << "{\n";
  s << "  \"result\": \"" << verdict_str() << "\"";
  if (!property_id.empty())
    s << ",\n  \"property\": \"" << json_escape(property_id) << "\"";
  if (!description.empty())
    s << ",\n  \"description\": \"" << json_escape(description) << "\"";
  if (!raw_output.empty())
    s << ",\n  \"raw_output\": \"" << json_escape(raw_output) << "\"";
  s << "\n}\n";
  return s.str();
}

// Locate the esbmc binary: the $ESBMC override, then alongside or relative to
// this executable (installed bin/ and the build tree), then $PATH.
static fs::path locate_esbmc()
{
  if (const char *env = std::getenv("ESBMC"))
    if (*env && fs::exists(env))
      return fs::path(env);

  boost::system::error_code ec;
  fs::path self = boost::dll::program_location(ec);
  if (!ec)
  {
    const fs::path dir = self.parent_path();
    const fs::path candidates[] = {
      dir / "esbmc",                                // installed: bin/esbmc
      dir / ".." / ".." / "src" / "esbmc" / "esbmc" // build tree
    };
    for (const auto &cand : candidates)
      if (fs::exists(cand))
        return cand;
  }

  return bp::search_path("esbmc");
}

// Run esbmc with the given arguments, returning its merged stdout+stderr.
static std::string
run_esbmc(const fs::path &esbmc, const std::vector<std::string> &args)
{
  bp::ipstream out;
  bp::child proc(esbmc, args, (bp::std_out & bp::std_err) > out);

  std::ostringstream captured;
  std::string line;
  while (std::getline(out, line))
    captured << line << '\n';
  proc.wait();
  return captured.str();
}

// Extract the human-readable description from esbmc's "Violated property:"
// block, which lists the source location ("file ... line ..."), the property
// comment (the description set by the property encoder), and the guard
// expression on consecutive lines.  The location line may be absent or the
// layout may shift, so take the first non-empty, non-location line as the
// description rather than indexing a fixed position.
static void
parse_violated_property(const std::string &output, LdVerifyResult &r)
{
  std::istringstream ss(output);
  std::string line;
  while (std::getline(ss, line) &&
         line.find("Violated property:") == std::string::npos)
    ;

  while (std::getline(ss, line))
  {
    const size_t start = line.find_first_not_of(" \t");
    if (start == std::string::npos)
      break; // a blank line ends the block
    const std::string trimmed = line.substr(start);
    if (trimmed.rfind("file ", 0) == 0)
      continue; // source-location line, not the description
    r.description = trimmed;
    return;
  }
}

// esbmc prints its verdict as a standalone, unindented line; match the whole
// line so that descriptive text echoed into the counterexample cannot be
// mistaken for a verdict.
static bool has_verdict_line(const std::string &output, const char *verdict)
{
  std::istringstream ss(output);
  std::string line;
  while (std::getline(ss, line))
  {
    if (!line.empty() && line.back() == '\r')
      line.pop_back(); // tolerate CRLF
    if (line == verdict)
      return true;
  }
  return false;
}

LdVerifyResult LdVerifyRunner::run(const LdVerifyOptions &opts)
{
  LdVerifyResult r;

  if (opts.program_path.empty())
  {
    r.verdict = LdVerifyResult::Verdict::Error;
    r.description = "no input program specified";
    return r;
  }

  const fs::path esbmc = locate_esbmc();
  if (esbmc.empty())
  {
    r.verdict = LdVerifyResult::Verdict::Error;
    r.description =
      "esbmc binary not found; set the ESBMC environment variable or add "
      "esbmc to PATH";
    return r;
  }

  // esbmc dispatches frontends by file extension, so a PLCopen .xml input must
  // be presented to it as a .ld file (plan §4.2).  Copy to a temp .ld when the
  // suffix is not already .ld.
  fs::path input = opts.program_path;
  fs::path temp_ld;

  // Remove the staged temp .ld on every exit path, including exceptions.
  struct temp_cleanup
  {
    const fs::path &path;
    ~temp_cleanup()
    {
      if (!path.empty())
      {
        boost::system::error_code ec;
        fs::remove(path, ec);
      }
    }
  } cleanup{temp_ld};

  if (input.extension() != ".ld")
  {
    boost::system::error_code ec;
    const fs::path tmp_dir = fs::temp_directory_path(ec);
    if (ec)
    {
      r.verdict = LdVerifyResult::Verdict::Error;
      r.description =
        "could not determine the temporary directory: " + ec.message();
      return r;
    }
    // unique_path guarantees a fresh name, so the plain copy_file overload
    // suffices (no overwrite option needed).
    temp_ld = tmp_dir / fs::unique_path("ld-verify-%%%%-%%%%.ld");
    fs::copy_file(input, temp_ld, ec);
    if (ec)
    {
      r.verdict = LdVerifyResult::Verdict::Error;
      r.description = "failed to stage input '" + opts.program_path +
                      "' as a .ld file: " + ec.message();
      return r;
    }
    input = temp_ld;
  }

  std::vector<std::string> args{input.string()};

  const std::string strategy =
    opts.strategy.empty() ? "k-induction" : opts.strategy;
  if (strategy == "k-induction")
  {
    args.push_back("--k-induction");
    args.push_back("--unlimited-k-steps");
  }
  else if (strategy == "bmc")
  {
    args.push_back("--incremental-bmc");
    args.push_back("--unwind");
    args.push_back(std::to_string(opts.bmc_unwind));
  }
  else
  {
    r.verdict = LdVerifyResult::Verdict::Error;
    r.description = "unsupported strategy '" + strategy +
                    "' (expected 'k-induction' or 'bmc')";
    return r;
  }

  if (!opts.props_path.empty())
  {
    args.push_back("--ld-props");
    args.push_back(opts.props_path);
  }
  if (opts.fault_injection)
    args.push_back("--ld-fault-injection");

  std::string output;
  try
  {
    output = run_esbmc(esbmc, args);
  }
  catch (const std::exception &e)
  {
    r.verdict = LdVerifyResult::Verdict::Error;
    r.description = std::string("failed to run esbmc: ") + e.what();
    return r;
  }

  if (has_verdict_line(output, "VERIFICATION SUCCESSFUL"))
    r.verdict = LdVerifyResult::Verdict::Safe;
  else if (has_verdict_line(output, "VERIFICATION FAILED"))
  {
    r.verdict = LdVerifyResult::Verdict::Violation;
    parse_violated_property(output, r);
    r.raw_output = output;
  }
  else if (has_verdict_line(output, "VERIFICATION UNKNOWN"))
  {
    // BMC is only complete up to its unwind bound; an "unknown" under
    // k-induction means the proof did not converge.
    r.verdict = (strategy == "bmc") ? LdVerifyResult::Verdict::Incomplete
                                    : LdVerifyResult::Verdict::Unknown;
    r.raw_output = output;
  }
  else
  {
    // No recognisable verdict: esbmc crashed, was killed, or rejected the input.
    r.verdict = LdVerifyResult::Verdict::Error;
    r.description = "esbmc produced no verdict";
    r.raw_output = output;
  }

  return r;
}
