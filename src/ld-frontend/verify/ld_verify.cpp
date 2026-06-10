#include <ld-frontend/verify/ld_verify.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <unistd.h>

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

// Single-quote a string for safe inclusion in a /bin/sh command line.
static std::string shell_quote(const std::string &s)
{
  std::string q = "'";
  for (char c : s)
    q += (c == '\'') ? std::string("'\\''") : std::string(1, c);
  q += "'";
  return q;
}

// Locate the esbmc binary: honour $ESBMC, else rely on $PATH.
static std::string esbmc_binary()
{
  const char *e = std::getenv("ESBMC");
  return (e && *e) ? std::string(e) : std::string("esbmc");
}

// Pull the violated-property description out of an ESBMC counterexample.
// The trace prints "Violated property:" then an indented "file <path>" line,
// then the assertion comment (the property description).  Scan for the first
// non-empty, non-"file" line after the header rather than assuming a fixed
// layout, so the field survives minor trace-format changes.
static std::string extract_violated_description(const std::string &output)
{
  auto pos = output.find("Violated property:");
  if (pos == std::string::npos)
    return {};

  std::istringstream is(output.substr(pos));
  std::string line;
  std::getline(is, line); // consume the "Violated property:" header
  while (std::getline(is, line))
  {
    auto begin = line.find_first_not_of(" \t");
    if (begin == std::string::npos)
      continue; // blank line
    std::string trimmed = line.substr(begin);
    if (trimmed.rfind("file ", 0) == 0)
      continue; // source-location line, not the description
    return trimmed;
  }
  return {};
}

LdVerifyResult LdVerifyRunner::run(const LdVerifyOptions &opts)
{
  namespace fs = std::filesystem;
  LdVerifyResult r;

  if (opts.program_path.empty())
  {
    r.verdict = LdVerifyResult::Verdict::Error;
    r.description = "no input program specified";
    return r;
  }

  // "portfolio" currently shares the k-induction invocation (the dedicated
  // portfolio orchestration is future work); reject anything else so a typo is
  // not silently treated as k-induction.
  const bool bmc = (opts.strategy == "bmc");
  if (!bmc && opts.strategy != "k-induction" && opts.strategy != "portfolio")
  {
    r.verdict = LdVerifyResult::Verdict::Error;
    r.description =
      "unknown strategy '" + opts.strategy + "' (expected k-induction|bmc)";
    return r;
  }

  // ESBMC dispatches frontends by file extension (§4.2), so a PLCopen .xml
  // file must be staged as a .ld copy before invocation.  A PID-tagged name
  // keeps concurrent ld-verify runs from clobbering each other's staging file.
  std::string program = opts.program_path;
  std::string staged;
  if (fs::path(program).extension() != ".ld")
  {
    std::error_code ec;
    fs::path tmp = fs::temp_directory_path() /
                   (fs::path(program).stem().string() + "-ldverify-" +
                    std::to_string(static_cast<long>(getpid())) + ".ld");
    fs::copy_file(program, tmp, fs::copy_options::overwrite_existing, ec);
    if (ec)
    {
      r.verdict = LdVerifyResult::Verdict::Error;
      r.description = "could not stage .ld copy: " + ec.message();
      return r;
    }
    staged = tmp.string();
    program = staged;
  }

  std::ostringstream cmd;
  cmd << shell_quote(esbmc_binary()) << ' ' << shell_quote(program);
  if (!opts.props_path.empty())
    cmd << " --ld-props " << shell_quote(opts.props_path);
  if (bmc)
    // The scan loop is while(true); without --no-unwinding-assertions the
    // unwinding assertion fires at the bound and masquerades as a property
    // violation.  Suppressing it gives genuine bounded semantics: a real
    // property violation within the bound still FAILS, while its absence maps
    // to INCOMPLETE (never SAFE) below.
    cmd << " --unwind " << opts.bmc_unwind << " --no-unwinding-assertions";
  else
    cmd << " --k-induction --unlimited-k-steps";
  cmd << " 2>&1";

  std::string output;
  FILE *pipe = popen(cmd.str().c_str(), "r");
  if (!pipe)
  {
    if (!staged.empty())
    {
      std::error_code ec;
      fs::remove(staged, ec);
    }
    r.verdict = LdVerifyResult::Verdict::Error;
    r.description = "failed to launch esbmc (set $ESBMC or add it to PATH)";
    return r;
  }

  std::array<char, 4096> buf;
  size_t n;
  while ((n = fread(buf.data(), 1, buf.size(), pipe)) > 0)
    output.append(buf.data(), n);
  int status = pclose(pipe);

  if (!staged.empty())
  {
    std::error_code ec;
    fs::remove(staged, ec);
  }

  r.raw_output = output;

  if (output.find("VERIFICATION FAILED") != std::string::npos)
  {
    r.verdict = LdVerifyResult::Verdict::Violation;
    r.description = extract_violated_description(output);
  }
  else if (output.find("VERIFICATION SUCCESSFUL") != std::string::npos)
  {
    // A bounded BMC pass proves safety only up to the unwind depth (§3.7).
    r.verdict =
      bmc ? LdVerifyResult::Verdict::Incomplete : LdVerifyResult::Verdict::Safe;
  }
  else if (status != 0)
  {
    // No verdict token and a non-zero exit means esbmc never ran or crashed
    // (e.g. missing binary, parse error); this is an error, not an
    // inconclusive run.
    r.verdict = LdVerifyResult::Verdict::Error;
    r.description = "esbmc produced no verdict (exit status " +
                    std::to_string(status) + "); set $ESBMC or check the input";
  }
  else
  {
    r.verdict = LdVerifyResult::Verdict::Unknown;
  }

  return r;
}
