#include <python-frontend/complex_handler_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_exception_handler.h>

#include <cctype>
#include <cstdlib>
#include <string>

namespace complex_utils
{

namespace
{

std::string trim(std::string s)
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

} // anonymous namespace

bool parse_complex_string(
  const std::string &raw,
  double &real_out,
  double &imag_out)
{
  std::string s = trim(raw);
  if (s.empty())
    return false;

  // Accept optional wrapping parentheses (possibly repeated).
  while (s.size() > 2 && s.front() == '(' && s.back() == ')')
    s = trim(s.substr(1, s.size() - 2));

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
