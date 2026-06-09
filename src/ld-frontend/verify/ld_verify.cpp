#include <ld-frontend/verify/ld_verify.h>
#include <sstream>

std::string LdVerifyResult::to_json() const
{
  std::ostringstream s;
  s << "{\n";

  auto verdict_str = [&]() -> std::string {
    switch (verdict)
    {
    case Verdict::Safe:       return "SAFE";
    case Verdict::Violation:  return "VIOLATION";
    case Verdict::Unknown:    return "UNKNOWN";
    case Verdict::Incomplete: return "INCOMPLETE";
    case Verdict::Error:      return "ERROR";
    }
    return "UNKNOWN";
  };

  s << "  \"result\": \"" << verdict_str() << "\",\n";
  if (!property_id.empty())
    s << "  \"property\": \"" << property_id << "\",\n";
  if (!description.empty())
    s << "  \"description\": \"" << description << "\",\n";

  s << "  \"raw_output\": \"(see esbmc output)\"\n";
  s << "}\n";
  return s.str();
}

LdVerifyResult LdVerifyRunner::run(const LdVerifyOptions &opts)
{
  // The ld_verify runner is invoked when ld-verify is used as a standalone
  // CLI.  When SAFE-LD is invoked through the normal ESBMC driver (i.e.
  // via ld_languaget::typecheck) the result comes from ESBMC's own output
  // routines.  This stub performs the pipeline steps that are independent
  // of the ESBMC driver.
  (void)opts;
  LdVerifyResult r;
  r.verdict = LdVerifyResult::Verdict::Unknown;
  r.description = "Use the ESBMC driver for full verification";
  return r;
}
