#include <c_string2expr.h>
#include <clang-c-frontend/typecast.h>

#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <util/std_expr.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/message.h>

expression_node::expression_node(
  std::string_view node_value,
  node_kind node_type)
  : value(node_value), kind(node_type), left(nullptr), right(nullptr)
{
}

const char *expression_node::kind_to_string(node_kind kind)
{
  switch (kind)
  {
  case node_kind::unknown:
    return "unknown";
  case node_kind::identifier:
    return "identifier";
  case node_kind::integer_literal:
    return "integer_literal";
  case node_kind::unary_plus:
    return "unary_plus";
  case node_kind::unary_minus:
    return "unary_minus";
  case node_kind::logical_not:
    return "logical_not";
  case node_kind::add:
    return "add";
  case node_kind::subtract:
    return "subtract";
  case node_kind::multiply:
    return "multiply";
  case node_kind::divide:
    return "divide";
  case node_kind::less:
    return "less";
  case node_kind::less_equal:
    return "less_equal";
  case node_kind::greater:
    return "greater";
  case node_kind::greater_equal:
    return "greater_equal";
  case node_kind::equal:
    return "equal";
  case node_kind::not_equal:
    return "not_equal";
  case node_kind::logical_and:
    return "logical_and";
  case node_kind::logical_or:
    return "logical_or";
  case node_kind::member:
    return "member";
  case node_kind::index:
    return "index";
  }

  return "unknown";
}

void expression_node::output(std::ostream &out) const
{
  // print the AST as a tree
  out << value << " <" << kind_to_string(kind) << ">\n";

  if (left)
    node_output(left, out, "", right);

  if (right)
    node_output(right, out, "", false);
}

void expression_node::dump() const
{
  // print the AST
  std::ostringstream oss;
  output(oss);
  log_status("{}", oss.str());
}

void expression_node::node_output(
  const expression_node *node,
  std::ostream &out,
  const std::string &prefix,
  bool has_next_sibling)
{
  out << prefix << (has_next_sibling ? "├── " : "└── ") << node->value << " <"
      << kind_to_string(node->kind) << ">\n";

  if (!node->left && !node->right)
    return;

  const std::string child_prefix =
    prefix + (has_next_sibling ? "│   " : "    ");

  if (node->left)
    node_output(node->left, out, child_prefix, node->right);

  if (node->right)
    node_output(node->right, out, child_prefix, false);
}

expression_parser::expression_parser() : input_(), position_(0)
{
  // create a reusable parser instance
  current_.type = token_type::end;
  current_.begin = 0;
  current_.length = 0;
}

bool expression_parser::parse(
  std::string_view expression,
  const expression_node *&root)
{
  // parse the input expression and return the AST root via root
  input_ = expression;
  position_ = 0;
  nodes_.clear();
  root = nullptr;

  if (next_token())
    return true;

  if (parse_expression(0, root))
    return true;

  if (current_.type != token_type::end)
  {
    log_error("unexpected trailing token at position {}", current_.begin);
    return true;
  }

  return false;
}

bool expression_parser::parse_expression(
  int minimum_precedence,
  const expression_node *&node)
{
  // parse a binary expression with precedence climbing
  if (parse_prefix(node))
    return true;

  while (true)
  {
    const int precedence = binary_precedence(current_.type);
    if (precedence < minimum_precedence)
      break;

    const token operator_token = current_;
    const expression_node::node_kind kind = binary_node_kind(current_.type);

    if (next_token())
      return true;

    const expression_node *right_node = nullptr;
    if (parse_expression(precedence + 1, right_node))
      return true;

    node = make_node(token_text(operator_token), kind, node, right_node);
  }

  return false;
}

bool expression_parser::parse_prefix(const expression_node *&node)
{
  // parse unary prefix operators such as !, - and +
  if (
    current_.type == token_type::logical_not ||
    current_.type == token_type::minus || current_.type == token_type::plus)
  {
    const token operator_token = current_;
    expression_node::node_kind kind;

    switch (current_.type)
    {
    case token_type::logical_not:
      kind = expression_node::node_kind::logical_not;
      break;
    case token_type::minus:
      kind = expression_node::node_kind::unary_minus;
      break;
    case token_type::plus:
      kind = expression_node::node_kind::unary_plus;
      break;
    default:
      break;
    }

    if (next_token())
      return true;

    // prefix operators bind tighter than all supported binary operators
    const expression_node *operand_node = nullptr;
    if (parse_expression(100, operand_node))
      return true;

    node = make_node(token_text(operator_token), kind, operand_node, nullptr);
    return false;
  }

  return parse_postfix(node);
}

bool expression_parser::parse_postfix(const expression_node *&node)
{
  // parse postfix operators such as ., -> and []
  if (parse_primary(node))
    return true;

  while (true)
  {
    if (current_.type == token_type::dot)
    {
      // parse member: x.y
      if (next_token())
        return true;

      if (current_.type != token_type::identifier)
      {
        log_error(
          "expected identifier after '.' at position {}", current_.begin);
        return true;
      }

      const token member_token = current_;
      if (next_token())
        return true;

      const expression_node *member_node = make_node(
        token_text(member_token), expression_node::node_kind::identifier);

      node =
        make_node(".", expression_node::node_kind::member, node, member_node);
      continue;
    }

    if (current_.type == token_type::arrow)
    {
      // parse member: x->y
      if (next_token())
        return true;

      if (current_.type != token_type::identifier)
      {
        log_error(
          "expected identifier after '->' at position {}", current_.begin);
        return true;
      }

      const token member_token = current_;
      if (next_token())
        return true;

      const expression_node *member_node = make_node(
        token_text(member_token), expression_node::node_kind::identifier);

      node =
        make_node("->", expression_node::node_kind::member, node, member_node);
      continue;
    }

    if (current_.type == token_type::left_bracket)
    {
      // parse index: x[0]
      if (next_token())
        return true;

      const expression_node *index_node = nullptr;
      if (parse_expression(0, index_node))
        return true;

      if (current_.type != token_type::right_bracket)
      {
        log_error("expected ']' at position {}", current_.begin);
        return true;
      }

      if (next_token())
        return true;

      node =
        make_node("[]", expression_node::node_kind::index, node, index_node);
      continue;
    }

    break;
  }

  return false;
}

bool expression_parser::parse_primary(const expression_node *&node)
{
  // parse an identifier, integer literal, or parenthesized expression
  if (current_.type == token_type::identifier)
  {
    const token tok = current_;

    if (next_token())
      return true;

    node = make_node(token_text(tok), expression_node::node_kind::identifier);
    return false;
  }

  if (current_.type == token_type::integer)
  {
    const token tok = current_;

    if (next_token())
      return true;

    node =
      make_node(token_text(tok), expression_node::node_kind::integer_literal);
    return false;
  }

  if (current_.type == token_type::left_paren)
  {
    if (next_token())
      return true;

    if (parse_expression(0, node))
      return true;

    if (current_.type != token_type::right_paren)
    {
      log_error("expected ')' at position {}", current_.begin);
      return true;
    }

    if (next_token())
      return true;

    return false;
  }

  log_error("expected primary expression at position {}", current_.begin);
  return true;
}

bool expression_parser::next_token()
{
  // read the next token from the input string
  while (position_ < input_.size() &&
         std::isspace(static_cast<unsigned char>(input_[position_])) != 0)
  {
    ++position_;
  }

  current_.begin = position_;
  current_.length = 0;

  if (position_ >= input_.size())
  {
    current_.type = token_type::end;
    return false;
  }

  const char current_char = input_[position_];

  // identifier: [a-zA-Z_][a-zA-Z0-9_]*
  if (
    std::isalpha(static_cast<unsigned char>(current_char)) != 0 ||
    current_char == '_')
  {
    const std::size_t start = position_++;
    while (position_ < input_.size())
    {
      const char ch = input_[position_];
      if (std::isalnum(static_cast<unsigned char>(ch)) == 0 && ch != '_')
        break;
      ++position_;
    }

    current_.type = token_type::identifier;
    current_.begin = start;
    current_.length = position_ - start;
    return false;
  }

  // integer: [0-9]+
  if (std::isdigit(static_cast<unsigned char>(current_char)) != 0)
  {
    const std::size_t start = position_++;
    while (position_ < input_.size() &&
           std::isdigit(static_cast<unsigned char>(input_[position_])) != 0)
    {
      ++position_;
    }

    // type: [0-9]U
    while (position_ < input_.size() &&
           (input_[position_] == 'u' || input_[position_] == 'U' ||
            input_[position_] == 'l' || input_[position_] == 'L'))
    {
      ++position_;
    }

    current_.type = token_type::integer;
    current_.begin = start;
    current_.length = position_ - start;
    return false;
  }

  // two-character operators
  if (position_ + 1 < input_.size())
  {
    const char next_char = input_[position_ + 1];

    switch (current_char)
    {
    case '<':
      if (next_char == '=')
      {
        current_.type = token_type::less_equal;
        current_.begin = position_;
        current_.length = 2;
        position_ += 2;
        return false;
      }
      break;

    case '>':
      if (next_char == '=')
      {
        current_.type = token_type::greater_equal;
        current_.begin = position_;
        current_.length = 2;
        position_ += 2;
        return false;
      }
      break;

    case '=':
      if (next_char == '=')
      {
        current_.type = token_type::equal_equal;
        current_.begin = position_;
        current_.length = 2;
        position_ += 2;
        return false;
      }
      break;

    case '!':
      if (next_char == '=')
      {
        current_.type = token_type::not_equal;
        current_.begin = position_;
        current_.length = 2;
        position_ += 2;
        return false;
      }
      break;

    case '&':
      if (next_char == '&')
      {
        current_.type = token_type::logical_and;
        current_.begin = position_;
        current_.length = 2;
        position_ += 2;
        return false;
      }
      break;

    case '|':
      if (next_char == '|')
      {
        current_.type = token_type::logical_or;
        current_.begin = position_;
        current_.length = 2;
        position_ += 2;
        return false;
      }
      break;

    case '-':
      if (next_char == '>')
      {
        current_.type = token_type::arrow;
        current_.begin = position_;
        current_.length = 2;
        position_ += 2;
        return false;
      }
      break;
    }
  }

  // one-character tokens
  ++position_;

  switch (current_char)
  {
  case '(':
    current_.type = token_type::left_paren;
    current_.length = 1;
    return false;

  case ')':
    current_.type = token_type::right_paren;
    current_.length = 1;
    return false;

  case '[':
    current_.type = token_type::left_bracket;
    current_.length = 1;
    return false;

  case ']':
    current_.type = token_type::right_bracket;
    current_.length = 1;
    return false;

  case '.':
    current_.type = token_type::dot;
    current_.length = 1;
    return false;

  case '+':
    current_.type = token_type::plus;
    current_.length = 1;
    return false;

  case '-':
    current_.type = token_type::minus;
    current_.length = 1;
    return false;

  case '*':
    current_.type = token_type::star;
    current_.length = 1;
    return false;

  case '/':
    current_.type = token_type::slash;
    current_.length = 1;
    return false;

  case '<':
    current_.type = token_type::less;
    current_.length = 1;
    return false;

  case '>':
    current_.type = token_type::greater;
    current_.length = 1;
    return false;

  case '!':
    current_.type = token_type::logical_not;
    current_.length = 1;
    return false;

  default:
    log_error(
      "unsupported character '{}' at position {}",
      current_char,
      current_.begin);
    current_.type = token_type::end;
    return true;
  }
}

int expression_parser::binary_precedence(token_type type) const
{
  // return precedence for a supported binary operator
  switch (type)
  {
  case token_type::logical_or:
    return 1;
  case token_type::logical_and:
    return 2;
  case token_type::equal_equal:
  case token_type::not_equal:
    return 3;
  case token_type::less:
  case token_type::less_equal:
  case token_type::greater:
  case token_type::greater_equal:
    return 4;
  case token_type::plus:
  case token_type::minus:
    return 5;
  case token_type::star:
  case token_type::slash:
    return 6;
  default:
    return -1;
  }
}

expression_node::node_kind
expression_parser::binary_node_kind(token_type type) const
{
  switch (type)
  {
  case token_type::plus:
    return expression_node::node_kind::add;
  case token_type::minus:
    return expression_node::node_kind::subtract;
  case token_type::star:
    return expression_node::node_kind::multiply;
  case token_type::slash:
    return expression_node::node_kind::divide;
  case token_type::less:
    return expression_node::node_kind::less;
  case token_type::less_equal:
    return expression_node::node_kind::less_equal;
  case token_type::greater:
    return expression_node::node_kind::greater;
  case token_type::greater_equal:
    return expression_node::node_kind::greater_equal;
  case token_type::equal_equal:
    return expression_node::node_kind::equal;
  case token_type::not_equal:
    return expression_node::node_kind::not_equal;
  case token_type::logical_and:
    return expression_node::node_kind::logical_and;
  case token_type::logical_or:
    return expression_node::node_kind::logical_or;
  default:
    return expression_node::node_kind::unknown;
  }
}

std::string_view expression_parser::token_text(const token &tok) const
{
  // return the original text slice for a token
  return input_.substr(tok.begin, tok.length);
}

const expression_node *expression_parser::make_node(
  std::string_view value,
  expression_node::node_kind kind,
  const expression_node *left,
  const expression_node *right)
{
  // allocate one AST node from the internal node pool
  nodes_.emplace_back(value, kind);
  expression_node &node = nodes_.back();
  node.left = left;
  node.right = right;
  return &node;
}

expression_converter::expression_converter(contextt &ns, locationt &location)
  : context_(ns), location_(location)
{
}

bool expression_converter::convert(const expression_node *root, exprt &expr)
{
  if (!root)
    return true;

  return get_expr(*root, expr);
}

static std::string to_lower_string(std::string_view text)
{
  std::string result;
  result.reserve(text.size());

  for (char ch : text)
  {
    result += static_cast<char>(
      std::tolower(static_cast<unsigned char>(ch)));
  }

  return result;
}

static bool build_integer_constant(std::string_view text, exprt &expr)
{
  std::size_t pos = 0;

  while (
    pos < text.size() &&
    std::isdigit(static_cast<unsigned char>(text[pos])) != 0)
  {
    ++pos;
  }

  if (pos == 0)
    return true;

  const std::string number_part(text.substr(0, pos));
  const std::string suffix = to_lower_string(text.substr(pos));

  typet type;

  if (suffix.empty())
  {
    type = int_type();
  }
  else if (suffix == "u")
  {
    type = uint_type();
  }
  else if (suffix == "l")
  {
    type = long_int_type();
  }
  else if (suffix == "ul" || suffix == "lu")
  {
    type = long_uint_type();
  }
  else if (suffix == "ll")
  {
    type = long_long_int_type();
  }
  else if (suffix == "ull" || suffix == "llu")
  {
    type = long_long_uint_type();
  }
  else
  {
    log_error("unsupport integer suffix: {}", suffix);
    return true;
  }

  expr = constant_exprt(string2integer(number_part), type);
  return false;
}

bool expression_converter::get_expr(const expression_node &node, exprt &expr)
{
  switch (node.kind)
  {
  case expression_node::node_kind::identifier:
  {
    std::list<exprt> matches;
    forall_symbol_base_map (
      it, context_.symbol_base_map, std::string(node.value))
    {
      const symbolt *s = context_.find_symbol(it->second);
      if (!s)
        continue;

      // Match the scope
      // case 1:
      // void func() {int x;} match the function name
      // case 2:
      // int x; void func() {} global var is match every func name
      if (
        (s->location.function() == location_.function()) ||
        (s->static_lifetime && s->location.function() == ""))
        matches.push_back(symbol_expr(*s));
    }

    if (matches.empty())
    {
      log_error("symbol `{}' not found", node.value);
      return true;
    }

    if (matches.size() > 1)
    {
      log_error("symbol `{}' is ambiguous", node.value);
      return true;
    }

    expr = matches.front();
    break;
  }

  case expression_node::node_kind::integer_literal:
  {
    if (build_integer_constant(node.value, expr))
      return true;

    break;
  }

  case expression_node::node_kind::unary_plus:
  {
    expr = exprt("unary+");
    exprt left;
    if (get_expr(*node.left, left))
      return true;

    expr.type() = left.type();
    expr.move_to_operands(left);
    break;
  }

  case expression_node::node_kind::unary_minus:
  {
    expr = exprt("unary-");
    exprt left;
    if (get_expr(*node.left, left))
      return true;

    expr.type() = left.type();
    expr.move_to_operands(left);
    break;
  }

  case expression_node::node_kind::logical_not:
  {
    expr = exprt("not", bool_type());
    exprt left;
    if (get_expr(*node.left, left))
      return true;

    expr.move_to_operands(left);
    break;
  }

  case expression_node::node_kind::add:
  {
    expr = exprt("+");
    exprt left;
    if (get_expr(*node.left, left))
      return true;
    exprt right;
    if (get_expr(*node.right, right))
      return true;

    gen_typecast_arithmetic(context_, left, right);
    expr.type() = left.type();
    expr.move_to_operands(left, right);
    break;
  }

  case expression_node::node_kind::subtract:
  {
    expr = exprt("-");
    exprt left;
    if (get_expr(*node.left, left))
      return true;
    exprt right;
    if (get_expr(*node.right, right))
      return true;

    gen_typecast_arithmetic(context_, left, right);
    expr.type() = left.type();
    expr.move_to_operands(left, right);
    break;
  }

  case expression_node::node_kind::multiply:
  {
    expr = exprt("*");
    exprt left;
    if (get_expr(*node.left, left))
      return true;
    exprt right;
    if (get_expr(*node.right, right))
      return true;

    gen_typecast_arithmetic(context_, left, right);
    expr.type() = left.type();
    expr.move_to_operands(left, right);
    break;
  }

  case expression_node::node_kind::divide:
  {
    expr = exprt("/");
    exprt left;
    if (get_expr(*node.left, left))
      return true;
    exprt right;
    if (get_expr(*node.right, right))
      return true;

    gen_typecast_arithmetic(context_, left, right);
    expr.type() = left.type();
    expr.move_to_operands(left, right);
    break;
  }

  case expression_node::node_kind::less:
  {
    expr = exprt("<", bool_type());
    exprt left;
    if (get_expr(*node.left, left))
      return true;
    exprt right;
    if (get_expr(*node.right, right))
      return true;

    gen_typecast_arithmetic(context_, left, right);
    expr.type() = left.type();
    expr.move_to_operands(left, right);
    break;
  }

  case expression_node::node_kind::less_equal:
  {
    expr = exprt("<=", bool_type());
    exprt left;
    if (get_expr(*node.left, left))
      return true;
    exprt right;
    if (get_expr(*node.right, right))
      return true;

    gen_typecast_arithmetic(context_, left, right);
    expr.move_to_operands(left, right);
    break;
  }

  case expression_node::node_kind::greater:
  {
    expr = exprt(">", bool_type());
    exprt left;
    if (get_expr(*node.left, left))
      return true;
    exprt right;
    if (get_expr(*node.right, right))
      return true;

    gen_typecast_arithmetic(context_, left, right);
    expr.move_to_operands(left, right);
    break;
  }

  case expression_node::node_kind::greater_equal:
  {
    expr = exprt(">=", bool_type());
    exprt left;
    if (get_expr(*node.left, left))
      return true;
    exprt right;
    if (get_expr(*node.right, right))
      return true;

    gen_typecast_arithmetic(context_, left, right);
    expr.move_to_operands(left, right);
    break;
  }

  case expression_node::node_kind::equal:
  {
    expr = exprt("=", bool_type());
    exprt left;
    if (get_expr(*node.left, left))
      return true;
    exprt right;
    if (get_expr(*node.right, right))
      return true;

    gen_typecast_arithmetic(context_, left, right);
    expr.move_to_operands(left, right);
    break;
  }

  case expression_node::node_kind::not_equal:
  {
    expr = exprt("notequal", bool_type());
    exprt left;
    if (get_expr(*node.left, left))
      return true;
    exprt right;
    if (get_expr(*node.right, right))
      return true;

    gen_typecast_arithmetic(context_, left, right);
    expr.move_to_operands(left, right);
    break;
  }

  case expression_node::node_kind::logical_and:
  {
    expr = exprt("and", bool_type());
    exprt left;
    if (get_expr(*node.left, left))
      return true;
    exprt right;
    if (get_expr(*node.right, right))
      return true;

    expr.move_to_operands(left, right);
    break;
  }

  case expression_node::node_kind::logical_or:
  {
    expr = exprt("or", bool_type());
    exprt left;
    if (get_expr(*node.left, left))
      return true;
    exprt right;
    if (get_expr(*node.right, right))
      return true;

    expr.move_to_operands(left, right);
    break;
  }

  case expression_node::node_kind::member:
  {
    exprt left;
    if (get_expr(*node.left, left))
      return true;

    if (!left.type().is_struct() && !left.type().is_union())
    {
      log_error("expect struct or union types");
      return true;
    }

    struct_union_typet type = to_struct_union_type(left.type());
    std::string comp = std::string(node.right->value);
    if (!type.has_component(comp))
    {
      log_error("struct or union types don't have component {}", comp);
      return true;
    }

    expr = member_exprt(left, comp, type.get_component(comp).type());
    break;
  }

  case expression_node::node_kind::index:
  {
    exprt left;
    if (get_expr(*node.left, left))
      return true;
    exprt right;
    if (get_expr(*node.right, right))
      return true;

    expr = exprt("index", left.type().subtype());
    expr.move_to_operands(left);
    expr.move_to_operands(right);
    break;
  }

  default:
    log_error("unsupported expression node type");
    return true;
  }

  return false;
}
