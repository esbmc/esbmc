#include <python-frontend/complex_handler_utils.h>
#include <python-frontend/convert_float_literal.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_exception_handler.h>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <string>

#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/ieee_float.h>

namespace complex_utils
{
namespace
{
std::string trim(const std::string &s)
{
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])))
    ++b;
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
    --e;
  return s.substr(b, e - b);
}

bool parse_double_strict(const std::string &s, double &out)
{
  if (s.empty())
    return false;
  char *end = nullptr;
  out = std::strtod(s.c_str(), &end);
  return end == s.c_str() + s.size();
}

// Convert a numeric string (possibly with leading '+') to an exprt constant
// using convert_float_literal, giving bit-exact consistency with Python AST
// float constants. Special values (inf/nan) fall back to from_double.
exprt build_float_exprt(const std::string &s)
{
  std::string normalized = s;
  if (!normalized.empty() && normalized.front() == '+')
    normalized.erase(0, 1);

  // convert_float_literal cannot parse "inf"/"nan" spellings.
  if (!normalized.empty())
  {
    char *end = nullptr;
    const double val = std::strtod(normalized.c_str(), &end);
    if (end == normalized.c_str() + normalized.size() && !std::isfinite(val))
      return from_double(val, double_type());
  }

  exprt dest;
  convert_float_literal(normalized, dest);
  return dest;
}

} // anonymous namespace

bool parse_complex_string(
  const std::string &raw,
  double &real_out,
  double &imag_out)
{
  std::string s = trim(raw);
  if (s.empty())
    return false;

  // Accept at most one pair of wrapping parentheses (CPython does not
  // allow nested parens, e.g. complex("((1+2j))") raises ValueError).
  int strip_count = 0;
  while (s.size() > 2 && s.front() == '(' && s.back() == ')' && strip_count < 1)
  {
    s = trim(s.substr(1, s.size() - 2));
    ++strip_count;
  }
  if (s.empty())
    return false;

  // CPython rejects any internal whitespace inside the numeric body
  // (only leading/trailing whitespace, plus whitespace immediately
  // inside a single pair of parentheses, is allowed).
  for (char c : s)
    if (std::isspace(static_cast<unsigned char>(c)))
      return false;

  auto lower_last = [](char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  };

  // No imaginary suffix: parse as real number.
  if (lower_last(s.back()) != 'j')
  {
    double r = 0.0;
    if (!parse_double_strict(s, r))
      return false;
    real_out = r;
    imag_out = 0.0;
    return true;
  }

  std::string body = trim(s.substr(0, s.size() - 1));
  if (body.empty() || body == "+" || body == "-")
  {
    real_out = 0.0;
    imag_out = (body == "-") ? -1.0 : 1.0;
    return true;
  }

  size_t split = std::string::npos;
  for (size_t i = 1; i < body.size(); ++i)
  {
    const char c = body[i];
    if (c == '+' || c == '-')
    {
      const char prev = body[i - 1];
      if (prev != 'e' && prev != 'E')
        split = i;
    }
  }

  if (split == std::string::npos)
  {
    double imag = 0.0;
    if (!parse_double_strict(body, imag))
      return false;
    real_out = 0.0;
    imag_out = imag;
    return true;
  }

  const std::string real_part = trim(body.substr(0, split));
  const std::string imag_part = trim(body.substr(split));
  double real = 0.0;
  if (!parse_double_strict(real_part, real))
    return false;

  double imag = 0.0;
  if (imag_part == "+" || imag_part == "-")
    imag = (imag_part == "-") ? -1.0 : 1.0;
  else if (!parse_double_strict(imag_part, imag))
    return false;

  real_out = real;
  imag_out = imag;
  return true;
}

bool parse_complex_string(
  const std::string &raw,
  exprt &real_out,
  exprt &imag_out)
{
  std::string s = trim(raw);
  if (s.empty())
    return false;

  int strip_count = 0;
  while (s.size() > 2 && s.front() == '(' && s.back() == ')' && strip_count < 1)
  {
    s = trim(s.substr(1, s.size() - 2));
    ++strip_count;
  }
  if (s.empty())
    return false;

  for (char c : s)
    if (std::isspace(static_cast<unsigned char>(c)))
      return false;

  auto lower_last = [](char c)
  { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); };

  if (lower_last(s.back()) != 'j')
  {
    char *end = nullptr;
    std::strtod(s.c_str(), &end);
    if (end != s.c_str() + s.size())
      return false;
    real_out = build_float_exprt(s);
    imag_out = build_float_exprt("0");
    return true;
  }

  std::string body = trim(s.substr(0, s.size() - 1));
  if (body.empty() || body == "+")
  {
    real_out = build_float_exprt("0");
    imag_out = build_float_exprt("1");
    return true;
  }
  if (body == "-")
  {
    real_out = build_float_exprt("0");
    imag_out = build_float_exprt("-1");
    return true;
  }

  size_t split = std::string::npos;
  for (size_t i = 1; i < body.size(); ++i)
  {
    const char c = body[i];
    if (c == '+' || c == '-')
    {
      const char prev = body[i - 1];
      if (prev != 'e' && prev != 'E')
        split = i;
    }
  }

  if (split == std::string::npos)
  {
    char *end = nullptr;
    std::strtod(body.c_str(), &end);
    if (end != body.c_str() + body.size())
      return false;
    real_out = build_float_exprt("0");
    imag_out = build_float_exprt(body);
    return true;
  }

  const std::string real_part = trim(body.substr(0, split));
  const std::string imag_part = trim(body.substr(split));

  if (!real_part.empty())
  {
    char *end = nullptr;
    std::strtod(real_part.c_str(), &end);
    if (end != real_part.c_str() + real_part.size())
      return false;
  }

  if (imag_part == "+" || imag_part == "-")
  {
    real_out = build_float_exprt(real_part.empty() ? "0" : real_part);
    imag_out = build_float_exprt(imag_part == "+" ? "1" : "-1");
    return true;
  }

  char *end = nullptr;
  std::strtod(imag_part.c_str(), &end);
  if (end != imag_part.c_str() + imag_part.size())
    return false;

  real_out = build_float_exprt(real_part.empty() ? "0" : real_part);
  imag_out = build_float_exprt(imag_part);
  return true;
}

exprt raise_math_real_type_error_expr(python_converter &converter)
{
  return converter.get_exception_handler().gen_exception_raise(
    "TypeError", "must be real number, not complex");
}

exprt raise_math_int_type_error_expr(python_converter &converter)
{
  return converter.get_exception_handler().gen_exception_raise(
    "TypeError", "'complex' object cannot be interpreted as an integer");
}

} // namespace complex_utils
