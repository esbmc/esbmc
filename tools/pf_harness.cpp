// Standalone diagnostic harness for py_percent_format (PR #5695 Linux CI repro).
// Extracts the exact helper functions from
// src/python-frontend/converter/converter_binop.cpp and exercises the cases
// from regression/python/percent_format_spec/main.py. Depends only on the
// standard library + nlohmann/json. Temporary debug artifact; not for merge.
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>
#include <cstdio>
#include <cctype>
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace {
bool py_const_int(const nlohmann::json &a, long long &out)
{
  if (!a.is_object())
    return false;
  const std::string t = a.value("_type", "");
  if (t == "Constant" && a.contains("value"))
  {
    if (a["value"].is_number_integer())
    {
      out = a["value"].get<long long>();
      return true;
    }
    if (a["value"].is_boolean())
    {
      out = a["value"].get<bool>() ? 1 : 0;
      return true;
    }
    return false;
  }
  if (
    t == "UnaryOp" && a.contains("operand") &&
    a.value("op", nlohmann::json::object()).value("_type", "") == "USub")
  {
    long long v;
    if (py_const_int(a["operand"], v))
    {
      out = -v;
      return true;
    }
  }
  return false;
}

// Parse a constant double from a Python AST argument node: a Constant float or
// integer/bool, or a UnaryOp(USub) over one. Returns false otherwise.
bool py_const_double(const nlohmann::json &a, double &out)
{
  if (!a.is_object())
    return false;
  const std::string t = a.value("_type", "");
  if (t == "Constant" && a.contains("value"))
  {
    if (a["value"].is_number_float())
    {
      out = a["value"].get<double>();
      return true;
    }
    if (a["value"].is_number_integer())
    {
      out = static_cast<double>(a["value"].get<long long>());
      return true;
    }
    if (a["value"].is_boolean())
    {
      out = a["value"].get<bool>() ? 1.0 : 0.0;
      return true;
    }
    return false;
  }
  if (
    t == "UnaryOp" && a.contains("operand") &&
    a.value("op", nlohmann::json::object()).value("_type", "") == "USub")
  {
    double v;
    if (py_const_double(a["operand"], v))
    {
      out = -v;
      return true;
    }
  }
  return false;
}

// Pad `body` to `width` per the % flags: '-' left-justifies; '0' zero-pads a
// numeric value (after any leading sign); otherwise space-pad on the left.
std::string pad_format_field(
  const std::string &body,
  int width,
  bool left,
  bool zero,
  bool numeric)
{
  if (static_cast<int>(body.size()) >= width)
    return body;
  const int n = width - static_cast<int>(body.size());
  if (left)
    return body + std::string(n, ' ');
  if (zero && numeric)
  {
    size_t sign =
      (!body.empty() && (body[0] == '-' || body[0] == '+' || body[0] == ' '))
        ? 1
        : 0;
    return body.substr(0, sign) + std::string(n, '0') + body.substr(sign);
  }
  return std::string(n, ' ') + body;
}

// Apply the +/space sign flags to a non-negative numeric string.
std::string apply_sign_flags(const std::string &s, bool plus, bool space)
{
  if (s.empty() || s[0] == '-')
    return s;
  if (plus)
    return "+" + s;
  if (space)
    return " " + s;
  return s;
}

// Constant-fold a printf-style ``str % args`` formatting, matching CPython for
// the supported conversions. Throws std::runtime_error for any unsupported
// conversion, flag/width/precision, or non-constant argument, so the caller
// surfaces a clean diagnostic instead of mis-lowering ``str % x`` to pointer
// arithmetic (which crashed the SMT backend, #5495).
std::string py_percent_format(
  const std::string &fmt,
  const std::vector<nlohmann::json> &args,
  const std::map<std::string, nlohmann::json> &mapping)
{
  std::string out;
  size_t argi = 0;
  // When a `%(name)` mapping key was just parsed, `forced` points at its value
  // node and next_arg() returns it (without consuming a positional argument).
  const nlohmann::json *forced = nullptr;
  auto next_arg = [&]() -> const nlohmann::json & {
    if (forced != nullptr)
      return *forced;
    if (argi >= args.size())
      throw std::runtime_error(
        "TypeError: not enough arguments for format string");
    return args[argi++];
  };

  for (size_t i = 0; i < fmt.size(); ++i)
  {
    if (fmt[i] != '%')
    {
      out.push_back(fmt[i]);
      continue;
    }
    if (i + 1 >= fmt.size())
      throw std::runtime_error("ValueError: incomplete format");

    // Mapping form: %(name)<conv> looks `name` up in the right-hand dict.
    // Per CPython's %[(key)][flags][width][.precision]conv grammar the key
    // comes first, before any flags/width/precision.
    forced = nullptr;
    if (fmt[i + 1] == '(')
    {
      const size_t close = fmt.find(')', i + 2);
      if (close == std::string::npos)
        throw std::runtime_error("ValueError: incomplete format key");
      const std::string key = fmt.substr(i + 2, close - (i + 2));
      const auto it = mapping.find(key);
      if (it == mapping.end())
        throw std::runtime_error(
          "KeyError: '" + key + "' in str % mapping (or non-constant key)");
      forced = &it->second;
      i = close; // advance to ')'; the conversion spec follows
      if (i + 1 >= fmt.size())
        throw std::runtime_error("ValueError: incomplete format");
    }

    // Parse an optional conversion spec: %[flags][width][.precision]<conv>.
    // Cap width/precision so a pathological digit run cannot overflow int
    // (signed-overflow UB); any field this wide is rejected below.
    constexpr int kMaxField = 1000000;
    bool left = false, zero = false, plus = false, space = false;
    for (; i + 1 < fmt.size(); ++i)
    {
      const char f = fmt[i + 1];
      // '#' (alternate form) is not rendered, so do not accept it as a flag —
      // it falls through to the unknown-conversion reject rather than being
      // silently dropped (which would miscompile e.g. '%#x').
      if (f == '-')
        left = true;
      else if (f == '0')
        zero = true;
      else if (f == '+')
        plus = true;
      else if (f == ' ')
        space = true;
      else
        break;
    }
    int width = 0;
    while (i + 1 < fmt.size() &&
           std::isdigit(static_cast<unsigned char>(fmt[i + 1])))
      width = std::min(kMaxField, width * 10 + (fmt[++i] - '0'));
    int prec = -1;
    if (i + 1 < fmt.size() && fmt[i + 1] == '.')
    {
      ++i;
      prec = 0;
      while (i + 1 < fmt.size() &&
             std::isdigit(static_cast<unsigned char>(fmt[i + 1])))
        prec = std::min(kMaxField, prec * 10 + (fmt[++i] - '0'));
    }
    if (width >= kMaxField || prec >= kMaxField)
      throw std::runtime_error(
        "unsupported: excessively large width/precision in str % formatting");
    if (i + 1 >= fmt.size())
      throw std::runtime_error("ValueError: incomplete format");
    const char c = fmt[++i];

    // Two-pass snprintf with a literal format (stays -Wformat-nonliteral-safe).
    auto fmt_double = [](char conv, int p, double d) -> std::string {
      std::string b;
      int n = 0;
      if (conv == 'f' || conv == 'F')
        n = std::snprintf(nullptr, 0, "%.*f", p, d);
      else if (conv == 'e' || conv == 'E')
        n = std::snprintf(nullptr, 0, "%.*e", p, d);
      else
        n = std::snprintf(nullptr, 0, "%.*g", p, d);
      if (n < 0)
        return b;
      b.resize(static_cast<size_t>(n));
      if (conv == 'f' || conv == 'F')
        std::snprintf(&b[0], static_cast<size_t>(n) + 1, "%.*f", p, d);
      else if (conv == 'e' || conv == 'E')
        std::snprintf(&b[0], static_cast<size_t>(n) + 1, "%.*e", p, d);
      else
        std::snprintf(&b[0], static_cast<size_t>(n) + 1, "%.*g", p, d);
      if (conv == 'F' || conv == 'E' || conv == 'G')
        for (char &ch : b)
          ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
      return b;
    };

    if (c == '%')
    {
      out.push_back('%');
      continue;
    }

    std::string body;
    bool numeric = false;
    // For integer conversions, precision is a minimum digit count (zero-filled
    // after any sign) — distinct from the field width.
    auto apply_int_precision = [&](std::string mag) -> std::string {
      if (prec >= 0 && static_cast<int>(mag.size()) < prec)
        mag = std::string(prec - mag.size(), '0') + mag;
      return mag;
    };

    if (c == 'd' || c == 'i' || c == 'u')
    {
      long long v;
      if (!py_const_int(next_arg(), v))
        throw std::runtime_error(
          "unsupported: non-constant argument in str % formatting");
      // Split the sign off so precision zero-fill lands after it; to_string
      // avoids the INT64_MIN negation overflow of -v.
      std::string s = std::to_string(v);
      const bool neg = !s.empty() && s[0] == '-';
      std::string mag = apply_int_precision(neg ? s.substr(1) : s);
      const std::string sgn = neg ? "-" : (plus ? "+" : (space ? " " : ""));
      body = sgn + mag;
      numeric = true;
    }
    else if (c == 'x' || c == 'X' || c == 'o')
    {
      long long v;
      if (!py_const_int(next_arg(), v))
        throw std::runtime_error(
          "unsupported: non-constant argument in str % formatting");
      if (v < 0)
        throw std::runtime_error(
          "unsupported: negative argument in str % %x/%o formatting");
      std::ostringstream ss;
      if (c == 'o')
        ss << std::oct << v;
      else
        ss << (c == 'X' ? std::uppercase : std::nouppercase) << std::hex << v;
      body = apply_int_precision(ss.str());
      numeric = true;
    }
    else if (
      c == 'f' || c == 'F' || c == 'e' || c == 'E' || c == 'g' || c == 'G')
    {
      double d;
      if (!py_const_double(next_arg(), d))
        throw std::runtime_error(
          "unsupported: non-constant argument in str % formatting");
      body =
        apply_sign_flags(fmt_double(c, prec >= 0 ? prec : 6, d), plus, space);
      numeric = true;
    }
    else if (c == 's')
    {
      const nlohmann::json &a = next_arg();
      long long v;
      if (
        a.value("_type", "") == "Constant" && a.contains("value") &&
        a["value"].is_boolean())
        // CPython renders bool via str(): "True"/"False", not "1"/"0".
        body = a["value"].get<bool>() ? "True" : "False";
      else if (
        a.value("_type", "") == "Constant" && a.contains("value") &&
        a["value"].is_string())
        body = a["value"].get<std::string>();
      else if (py_const_int(a, v))
        body = std::to_string(v);
      else
        throw std::runtime_error(
          "unsupported: non-constant argument in str % formatting");
      // %.Ns truncates the string to N characters.
      if (prec >= 0 && static_cast<int>(body.size()) > prec)
        body.resize(static_cast<size_t>(prec));
    }
    else if (c == 'c')
    {
      const nlohmann::json &a = next_arg();
      long long v;
      if (py_const_int(a, v))
      {
        if (v < 0 || v > 255)
          throw std::runtime_error(
            "unsupported: %c code points above 255 (non-ASCII) not modelled");
        body = std::string(1, static_cast<char>(v));
      }
      else if (
        a.value("_type", "") == "Constant" && a.contains("value") &&
        a["value"].is_string() && a["value"].get<std::string>().size() == 1)
        body = a["value"].get<std::string>();
      else
        throw std::runtime_error(
          "unsupported: %c argument in str % formatting");
    }
    else
      throw std::runtime_error(
        std::string("unsupported conversion '%") + c + "' in str % formatting");

    out += pad_format_field(std::move(body), width, left, zero, numeric);
  }

  if (argi != args.size())
    throw std::runtime_error(
      "TypeError: not all arguments converted during string formatting");
  return out;
}
} // namespace

static void dump_bytes(const std::string& s) {
  std::printf("  bytes=[");
  for (size_t i=0;i<s.size();++i) std::printf("%s%d", i?",":"", (int)(unsigned char)s[i]);
  std::printf("]\n");
}

int main() {
  struct Case { const char* fmt; nlohmann::json arg; const char* expect; };
  auto fnode = [](double v){ nlohmann::json j; j["_type"]="Constant"; j["value"]=v; return j; };
  auto inode = [](long long v){ nlohmann::json j; j["_type"]="Constant"; j["value"]=v; return j; };
  std::vector<Case> cases = {
    {"%.2f", fnode(3.14159), "3.14"},
    {"%.1f", fnode(3.14159), "3.1"},
    {"%.3f", fnode(3.14159), "3.142"},
    {"%f",   fnode(3.14),    "3.140000"},
    {"%8.2f",fnode(3.14159), "    3.14"},
    {"%07.2f",fnode(-3.1),   "-003.10"},
    {"%e",   fnode(1000.0),  "1.000000e+03"},
    {"%5d",  inode(42),      "   42"},
  };
  int fails = 0;
  std::printf("g++/clang build; testing py_percent_format on this libc/libstdc++\n");
  for (auto& c : cases) {
    std::vector<nlohmann::json> args; args.push_back(c.arg);
    std::map<std::string, nlohmann::json> mapping;
    std::string got = py_percent_format(c.fmt, args, mapping);
    bool ok = (got == c.expect);
    std::printf("[%s] fmt=%-7s got=\"%s\" (len=%zu) expect=\"%s\"\n",
                ok?"OK":"FAIL", c.fmt, got.c_str(), got.size(), c.expect);
    if (!ok) { dump_bytes(got); fails++; }
  }
  std::printf("\n%d failure(s)\n", fails);
  return fails ? 1 : 0;
}
