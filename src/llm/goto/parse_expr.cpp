#include <llm/goto/parse_expr.h>

#include <irep2/irep2_utils.h>
#include <util/arith/mp_arith.h>
#include <util/config/config.h>
#include <util/lang/c_typecast.h>
#include <util/lang/c_types.h>
#include <util/message/message.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <sstream>
#include <map>
#include <set>
#include <vector>

namespace llm
{
namespace
{
struct parse_errort
{
  std::string message;
  std::size_t offset;
};

struct tokent
{
  enum kindt
  {
    END,
    IDENTIFIER,
    INTEGER,
    PUNCTUATOR
  } kind;
  std::string text;
  std::size_t offset;
};

std::size_t punctuator_length(const std::string &s, std::size_t i)
{
  static const char *const punctuators[] = {
    "->", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "+",
    "-",  "*",  "/",  "%",  "<",  ">",  "!",  "~",  "&",  "|",
    "^",  "?",  ":",  "(",  ")",  "[",  "]",  "."};
  for (const char *p : punctuators)
    if (s.compare(i, std::char_traits<char>::length(p), p) == 0)
      return std::char_traits<char>::length(p);
  throw parse_errort{std::string("unexpected character '") + s[i] + "'", i};
}

std::vector<tokent> tokenize(const std::string &s)
{
  auto span = [&](std::size_t i, auto in_token) {
    while (i < s.size() && in_token(static_cast<unsigned char>(s[i])))
      ++i;
    return i;
  };
  auto word = [](unsigned char c) { return std::isalnum(c) || c == '_'; };

  std::vector<tokent> tokens;
  for (std::size_t i = span(0, isspace); i < s.size(); i = span(i, isspace))
  {
    const unsigned char c = s[i];
    const std::size_t start = i;
    tokent::kindt kind = tokent::PUNCTUATOR;
    if (std::isalpha(c) || c == '_')
    {
      kind = tokent::IDENTIFIER;
      i = span(i, word);
    }
    else if (std::isdigit(c))
    {
      kind = tokent::INTEGER;
      i = span(i, isalnum);
      if (i < s.size() && s[i] == '.')
        throw parse_errort{"floating-point literals are not supported", start};
    }
    else
      i += punctuator_length(s, i);
    tokens.push_back({kind, s.substr(start, i - start), start});
  }
  tokens.push_back({tokent::END, "", s.size()});
  return tokens;
}

const std::set<std::string> &integer_suffixes()
{
  static const std::set<std::string> suffixes = [] {
    std::set<std::string> out;
    for (const char *u : {"", "u", "U"})
      for (const char *l : {"", "l", "L", "ll", "LL"})
      {
        out.insert(std::string(u) + l);
        out.insert(std::string(l) + u);
      }
    return out;
  }();
  return suffixes;
}

std::vector<type2tc>
literal_candidates(bool is_unsigned, std::size_t longs, unsigned base)
{
  std::vector<type2tc> candidates;
  auto add = [&](const type2tc &s, const type2tc &u) {
    if (!is_unsigned)
      candidates.push_back(s);
    if (is_unsigned || base != 10)
      candidates.push_back(u);
  };
  if (longs == 0)
    add(int_type2(), uint_type2());
  if (longs <= 1)
    add(long_int_type2(), long_uint_type2());
  add(long_long_int_type2(), long_long_uint_type2());
  return candidates;
}

unsigned literal_base(std::string &digits)
{
  if (digits.size() < 2 || digits[0] != '0')
    return 10;
  if (digits[1] != 'x' && digits[1] != 'X')
    return 8;
  digits.erase(0, 2);
  return 16;
}

std::size_t literal_longs(const std::string &suffix)
{
  if (suffix.find_first_of("lL") == std::string::npos)
    return 0;
  const bool two = suffix.find("ll") != std::string::npos ||
                   suffix.find("LL") != std::string::npos;
  return two ? 2 : 1;
}

expr2tc integer_literal(const std::string &text, std::size_t offset)
{
  const std::size_t digits_end =
    std::min(text.find_first_of("uUlL"), text.size());
  std::string digits = text.substr(0, digits_end);
  const std::string suffix = text.substr(digits_end);
  if (!integer_suffixes().count(suffix))
    throw parse_errort{"malformed integer suffix '" + text + "'", offset};

  const unsigned base = literal_base(digits);
  const char *valid = base == 16  ? "0123456789abcdefABCDEF"
                      : base == 8 ? "01234567"
                                  : "0123456789";
  if (digits.empty() || digits.find_first_not_of(valid) != std::string::npos)
    throw parse_errort{"malformed integer literal '" + text + "'", offset};

  const bool is_unsigned = suffix.find_first_of("uU") != std::string::npos;
  const BigInt value = string2integer(digits, base);
  for (const type2tc &t :
       literal_candidates(is_unsigned, literal_longs(suffix), base))
    if (value < BigInt::power2(t->get_width() - is_signedbv_type(t)))
      return constant_int2tc(t, value);
  throw parse_errort{"integer literal '" + text + "' is too large", offset};
}

bool is_integer(const expr2tc &e)
{
  return is_bv_type(e) || is_bool_type(e);
}

bool is_arithmetic(const expr2tc &e)
{
  return is_integer(e) || is_fractional_type(e->type);
}

bool is_null_constant(const expr2tc &e)
{
  return (is_constant_int2t(e) && to_constant_int2t(e).value.is_zero()) ||
         (is_symbol2t(e) && to_symbol2t(e).thename == "NULL");
}

/// Type specifiers are unordered.
std::string type_key(std::vector<std::string> words)
{
  std::sort(words.begin(), words.end());
  std::string key;
  for (const std::string &w : words)
    key += (key.empty() ? "" : " ") + w;
  return key;
}

const std::map<std::string, std::function<type2tc()>> &cast_types()
{
  static const auto types = [] {
    std::map<std::string, std::function<type2tc()>> out;
    auto add = [&](const std::string &spelling, std::function<type2tc()> f) {
      std::vector<std::string> words;
      std::istringstream in(spelling);
      for (std::string w; in >> w;)
        words.push_back(w);
      out.emplace(type_key(words), f);
    };
    auto add_integer = [&](
                         const std::string &base,
                         std::function<type2tc()> s,
                         std::function<type2tc()> u) {
      const std::string with_int = base.empty() ? "int" : base + " int";
      for (const std::string &t : {with_int, "signed " + with_int})
        add(t, s);
      if (!base.empty())
        add(base, s);
      add("signed " + base, s);
      add("unsigned " + base, u);
      add("unsigned " + with_int, u);
    };
    const auto &c = config.ansi_c;
    add("_Bool", get_bool_type);
    add("bool", get_bool_type);
    add("char", char_type2);
    add("signed char", [&c] { return signedbv_type2tc(c.char_width); });
    add("unsigned char", [&c] { return unsignedbv_type2tc(c.char_width); });
    add_integer(
      "short",
      [&c] { return signedbv_type2tc(c.short_int_width); },
      [&c] { return unsignedbv_type2tc(c.short_int_width); });
    add_integer("", int_type2, uint_type2);
    add_integer("long", long_int_type2, long_uint_type2);
    add_integer("long long", long_long_int_type2, long_long_uint_type2);
    return out;
  }();
  return types;
}

class parsert
{
public:
  parsert(const std::string &text, const scopet &scope, const namespacet &ns)
    : tokens(tokenize(text)), scope(scope), ns(ns)
  {
  }

  expr2tc parse()
  {
    expr2tc e = conditional();
    if (peek().kind != tokent::END)
      fail("unexpected '" + peek().text + "'");
    return e;
  }

private:
  std::vector<tokent> tokens;
  std::size_t pos = 0;
  unsigned depth = 0;
  const scopet &scope;
  const namespacet &ns;

  const tokent &peek(std::size_t ahead = 0) const
  {
    return tokens[std::min(pos + ahead, tokens.size() - 1)];
  }

  bool accept(const char *punctuator)
  {
    if (peek().kind != tokent::PUNCTUATOR || peek().text != punctuator)
      return false;
    ++pos;
    return true;
  }

  void expect(const char *punctuator)
  {
    if (!accept(punctuator))
      fail(std::string("expected '") + punctuator + "'");
  }

  [[noreturn]] void fail(const std::string &message) const
  {
    throw parse_errort{message, peek().offset};
  }

  [[noreturn]] void fail(const std::string &message, std::size_t offset) const
  {
    throw parse_errort{message, offset};
  }

  struct nestingt
  {
    parsert &p;
    explicit nestingt(parsert &p) : p(p)
    {
      if (++p.depth > 256)
        p.fail("expression nested too deeply");
    }
    ~nestingt()
    {
      --p.depth;
    }
  };

  expr2tc conditional()
  {
    nestingt nesting(*this);
    expr2tc cond = binary(1);
    const std::size_t offset = peek().offset;
    if (!accept("?"))
      return cond;
    expr2tc then = conditional();
    expect(":");
    expr2tc otherwise = conditional();
    match_operands(then, otherwise, offset);
    return if2tc(then->type, as_bool(cond, offset), then, otherwise);
  }

  static int precedence(const tokent &t)
  {
    if (t.kind != tokent::PUNCTUATOR)
      return 0;
    static const std::map<std::string, int> table = {
      {"||", 1},
      {"&&", 2},
      {"|", 3},
      {"^", 4},
      {"&", 5},
      {"==", 6},
      {"!=", 6},
      {"<", 7},
      {"<=", 7},
      {">", 7},
      {">=", 7},
      {"<<", 8},
      {">>", 8},
      {"+", 9},
      {"-", 9},
      {"*", 10},
      {"/", 10},
      {"%", 10}};
    auto it = table.find(t.text);
    return it == table.end() ? 0 : it->second;
  }

  expr2tc binary(int min_precedence)
  {
    expr2tc lhs = unary();
    for (int p = precedence(peek()); p >= min_precedence;
         p = precedence(peek()))
    {
      const tokent op = peek();
      ++pos;
      expr2tc rhs = binary(p + 1);
      lhs = make_binary(op, lhs, rhs);
    }
    return lhs;
  }

  expr2tc unary()
  {
    nestingt nesting(*this);
    const std::size_t offset = peek().offset;
    if (accept("!"))
      return not2tc(as_bool(unary(), offset));
    if (accept("-"))
      return negation(unary(), offset);
    if (accept("+"))
      return promoted(unary(), offset);
    if (accept("~"))
    {
      expr2tc e = promoted(unary(), offset);
      if (!is_bv_type(e))
        fail("'~' needs an integer operand", offset);
      return bitnot2tc(e->type, e);
    }
    if (accept("*"))
      return dereference(unary(), offset);
    if (accept("&"))
      return address_of(unary(), offset);
    if (is_cast())
      return cast(offset);
    return postfix();
  }

  expr2tc negation(const expr2tc &operand, std::size_t offset) const
  {
    expr2tc e = promoted(operand, offset);
    if (is_constant_int2t(e) && is_signedbv_type(e))
      return constant_int2tc(e->type, -to_constant_int2t(e).value);
    return neg2tc(e->type, e);
  }

  expr2tc address_of(const expr2tc &e, std::size_t offset) const
  {
    if (
      !is_symbol2t(e) && !is_index2t(e) && !is_member2t(e) &&
      !is_dereference2t(e))
      fail("'&' needs an lvalue", offset);
    return address_of2tc(e->type, e);
  }

  expr2tc cast(std::size_t offset)
  {
    expect("(");
    const type2tc type = type_name();
    expect(")");
    expr2tc e = unary();
    if (!is_arithmetic(e) && !is_pointer_type(e))
      fail("cannot cast a non-scalar", offset);
    return e->type == type ? e : typecast2tc(type, e);
  }

  expr2tc postfix()
  {
    expr2tc e = primary();
    for (;;)
    {
      const std::size_t offset = peek().offset;
      if (accept("["))
      {
        expr2tc index = promoted(conditional(), offset);
        expect("]");
        const type2tc type = followed(e, offset);
        if (!is_bv_type(index))
          fail("array index is not an integer", offset);
        if (is_array_type(type))
          e = index2tc(to_array_type(type).subtype, e, index);
        else if (
          is_pointer_type(type) &&
          !is_empty_type(to_pointer_type(type).subtype))
          e = index2tc(to_pointer_type(type).subtype, e, index);
        else
          fail("subscript of a non-array", offset);
      }
      else if (accept("."))
        e = member(e, offset);
      else if (accept("->"))
        e = member(dereference(e, offset), offset);
      else
        return e;
    }
  }

  expr2tc primary()
  {
    const tokent t = peek();
    if (accept("("))
    {
      expr2tc e = conditional();
      expect(")");
      return e;
    }
    if (t.kind == tokent::INTEGER)
    {
      ++pos;
      return integer_literal(t.text, t.offset);
    }
    if (t.kind != tokent::IDENTIFIER)
      fail(
        t.kind == tokent::END ? "unexpected end of expression"
                              : "unexpected '" + t.text + "'");
    ++pos;
    if (t.text == "true")
      return gen_true_expr();
    if (t.text == "false")
      return gen_false_expr();
    auto it = scope.find(t.text);
    if (it != scope.end())
      return it->second;
    if (t.text == "NULL")
      return gen_zero(pointer_type2tc(get_empty_type()));
    fail("unknown identifier '" + t.text + "'", t.offset);
  }

  bool is_cast() const
  {
    static const std::set<std::string> keywords = {
      "_Bool", "bool", "char", "short", "int", "long", "signed", "unsigned"};
    return peek().kind == tokent::PUNCTUATOR && peek().text == "(" &&
           peek(1).kind == tokent::IDENTIFIER && keywords.count(peek(1).text) &&
           !scope.count(peek(1).text);
  }

  type2tc type_name()
  {
    const std::size_t offset = peek().offset;
    std::vector<std::string> words;
    while (peek().kind == tokent::IDENTIFIER)
      words.push_back(tokens[pos++].text);
    const auto &types = cast_types();
    auto it = types.find(type_key(std::move(words)));
    if (it == types.end())
      fail("unsupported type in cast", offset);
    return it->second();
  }

  type2tc followed(const expr2tc &e, std::size_t offset) const
  {
    if (
      is_symbol_type(e->type) &&
      !ns.lookup(to_symbol_type(e->type).symbol_name))
      fail("incomplete type", offset);
    return ns.follow(e->type);
  }

  expr2tc as_bool(const expr2tc &e, std::size_t offset) const
  {
    if (is_bool_type(e))
      return e;
    if (is_arithmetic(e) || is_pointer_type(e))
      return notequal2tc(e, gen_zero(e->type));
    fail("expected a scalar operand", offset);
  }

  expr2tc promoted(expr2tc e, std::size_t offset) const
  {
    if (!is_arithmetic(e))
      fail("expected an arithmetic operand", offset);
    c_typecastt(ns).implicit_typecast_arithmetic(e);
    return e;
  }

  void convert_arithmetic(expr2tc &a, expr2tc &b, std::size_t offset) const
  {
    if (!is_arithmetic(a) || !is_arithmetic(b))
      fail("expected arithmetic operands", offset);
    c_typecastt(ns).implicit_typecast_arithmetic(a, b);
  }

  void match_operands(expr2tc &a, expr2tc &b, std::size_t offset) const
  {
    if (is_pointer_type(a) && is_null_constant(b))
      b = gen_zero(a->type);
    else if (is_pointer_type(b) && is_null_constant(a))
      a = gen_zero(b->type);
    else if (is_arithmetic(a) && is_arithmetic(b))
      convert_arithmetic(a, b, offset);
    else if (a->type != b->type)
      fail("operand types do not match", offset);
  }

  expr2tc dereference(const expr2tc &e, std::size_t offset) const
  {
    const type2tc type = followed(e, offset);
    if (!is_pointer_type(type) || is_empty_type(to_pointer_type(type).subtype))
      fail("dereference of a non-pointer", offset);
    return dereference2tc(to_pointer_type(type).subtype, e);
  }

  expr2tc member(const expr2tc &e, std::size_t offset)
  {
    const tokent field = peek();
    if (field.kind != tokent::IDENTIFIER)
      fail("expected a member name");
    ++pos;
    const type2tc type = followed(e, offset);
    if (!is_struct_type(type) && !is_union_type(type))
      fail("member access on a non-struct", offset);
    const auto n = struct_union_get_component_number(type, field.text);
    if (!n)
      fail("no member named '" + field.text + "'", field.offset);
    return member2tc(struct_union_members(type)[*n], e, field.text);
  }

  expr2tc make_binary(const tokent &op, expr2tc a, expr2tc b) const
  {
    const std::string &o = op.text;
    const std::size_t offset = op.offset;
    if (o == "&&")
      return and2tc(as_bool(a, offset), as_bool(b, offset));
    if (o == "||")
      return or2tc(as_bool(a, offset), as_bool(b, offset));
    static const std::set<std::string> comparisons = {
      "==", "!=", "<", "<=", ">", ">="};
    if (comparisons.count(o))
      return comparison(o, a, b, offset);
    if ((o == "+" || o == "-") && (is_pointer_type(a) || is_pointer_type(b)))
      return pointer_arithmetic(o, a, b, offset);
    if (o == "<<" || o == ">>")
      return shift(o, a, b, offset);

    convert_arithmetic(a, b, offset);
    if (is_fractional_type(a->type))
      fail("floating-point arithmetic is not supported", offset);
    const expr2tc e = arithmetic(o, a, b);
    if (is_constant_int2t(a) && is_constant_int2t(b))
      if (expr2tc folded = e->simplify(); !is_nil_expr(folded))
        return folded;
    return e;
  }

  expr2tc
  comparison(const std::string &o, expr2tc a, expr2tc b, std::size_t offset)
    const
  {
    match_operands(a, b, offset);
    if (!is_arithmetic(a) && !is_pointer_type(a))
      fail("cannot compare non-scalars", offset);
    if (o == "==")
      return equality2tc(a, b);
    if (o == "!=")
      return notequal2tc(a, b);
    if (o == "<")
      return lessthan2tc(a, b);
    if (o == "<=")
      return lessthanequal2tc(a, b);
    if (o == ">")
      return greaterthan2tc(a, b);
    return greaterthanequal2tc(a, b);
  }

  expr2tc
  shift(const std::string &o, expr2tc a, expr2tc b, std::size_t offset) const
  {
    a = promoted(a, offset);
    b = promoted(b, offset);
    if (!is_bv_type(a) || !is_bv_type(b))
      fail("shift needs integer operands", offset);
    if (o == "<<")
      return shl2tc(a->type, a, b);
    if (is_signedbv_type(a))
      return ashr2tc(a->type, a, b);
    return lshr2tc(a->type, a, b);
  }

  static expr2tc
  arithmetic(const std::string &o, const expr2tc &a, const expr2tc &b)
  {
    const type2tc &t = a->type;
    if (o == "+")
      return add2tc(t, a, b);
    if (o == "-")
      return sub2tc(t, a, b);
    if (o == "*")
      return mul2tc(t, a, b);
    if (o == "/")
      return div2tc(t, a, b);
    if (o == "%")
      return modulus2tc(t, a, b);
    if (o == "&")
      return bitand2tc(t, a, b);
    if (o == "|")
      return bitor2tc(t, a, b);
    return bitxor2tc(t, a, b);
  }

  expr2tc pointer_arithmetic(
    const std::string &o,
    expr2tc a,
    expr2tc b,
    std::size_t offset) const
  {
    if (o == "+" && !is_pointer_type(a))
      std::swap(a, b);
    if (!is_pointer_type(a) || !is_integer(b))
      fail("pointer arithmetic needs a pointer and an integer", offset);
    b = promoted(b, offset);
    if (o == "+")
      return add2tc(a->type, a, b);
    return sub2tc(a->type, a, b);
  }
};
} // namespace

std::optional<expr2tc>
parse_expr(const std::string &text, const scopet &scope, const namespacet &ns)
{
  try
  {
    return parsert(text, scope, ns).parse();
  }
  catch (const parse_errort &e)
  {
    log_debug(
      "llm", "cannot parse '{}': {} at offset {}", text, e.message, e.offset);
    return std::nullopt;
  }
}
} // namespace llm
