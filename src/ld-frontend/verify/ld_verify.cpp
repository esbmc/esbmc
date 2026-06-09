#include <ld-frontend/verify/ld_verify.h>
#include <sstream>

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

LdVerifyResult LdVerifyRunner::run(const LdVerifyOptions &opts)
{
  (void)opts;
  LdVerifyResult r;
  r.verdict = LdVerifyResult::Verdict::Unknown;
  r.description = "Use the ESBMC driver for full verification";
  return r;
}
