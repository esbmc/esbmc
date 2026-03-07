#include <c_string2expr.h>

#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>

// Create a node with the given text.
expression_node::expression_node(std::string_view node_value)
  : value(node_value), left(nullptr), right(nullptr)
{
}

// Print the AST as a tree.
void expression_node::print(std::ostream &output_stream) const
{
  output_stream << value << '\n';

  if (left != nullptr)
    print_impl(left, output_stream, "", right != nullptr);

  if (right != nullptr)
    print_impl(right, output_stream, "", false);
}

// Print the AST to standard output.
void expression_node::print() const
{
  print(std::cout);
}

// Recursive helper used by print().
void expression_node::print_impl(
  const expression_node *node,
  std::ostream &output_stream,
  const std::string &prefix,
  bool has_next_sibling)
{
  output_stream << prefix << (has_next_sibling ? "├── " : "└── ") << node->value
                << '\n';

  const bool has_left = (node->left != nullptr);
  const bool has_right = (node->right != nullptr);

  if (!has_left && !has_right)
    return;

  const std::string child_prefix =
    prefix + (has_next_sibling ? "│   " : "    ");

  if (has_left)
    print_impl(node->left, output_stream, child_prefix, has_right);

  if (has_right)
    print_impl(node->right, output_stream, child_prefix, false);
}

// Create a reusable parser instance.
expression_parser::expression_parser() : input_(), position_(0)
{
  current_.type = token_type::end;
  current_.begin = 0;
  current_.length = 0;
}

// Parse the input expression and return the AST root.
const expression_node *expression_parser::parse(std::string_view expression)
{
  input_ = expression;
  position_ = 0;
  nodes_.clear();

  next_token();

  const expression_node *root = parse_expression(0);

  if (current_.type != token_type::end)
    throw_error("unexpected trailing token", current_.begin);

  return root;
}

// Parse a binary expression with precedence climbing.
const expression_node *
expression_parser::parse_expression(int minimum_precedence)
{
  const expression_node *left_node = parse_prefix();

  while (true)
  {
    const int precedence = binary_precedence(current_.type);
    if (precedence < minimum_precedence)
      break;

    const token operator_token = current_;
    next_token();

    const expression_node *right_node = parse_expression(precedence + 1);
    left_node = make_node(token_text(operator_token), left_node, right_node);
  }

  return left_node;
}

// Parse unary prefix operators such as !, - and +.
const expression_node *expression_parser::parse_prefix()
{
  if (
    current_.type == token_type::logical_not ||
    current_.type == token_type::minus || current_.type == token_type::plus)
  {
    const token operator_token = current_;
    next_token();

    // Prefix operators bind tighter than all supported binary operators.
    const expression_node *operand_node = parse_expression(100);
    return make_node(token_text(operator_token), operand_node, nullptr);
  }

  return parse_primary();
}

// Parse an identifier, integer literal, or parenthesized expression.
const expression_node *expression_parser::parse_primary()
{
  if (
    current_.type == token_type::identifier ||
    current_.type == token_type::integer)
  {
    const token tok = current_;
    next_token();
    return make_node(token_text(tok));
  }

  if (current_.type == token_type::left_paren)
  {
    next_token();

    const expression_node *node = parse_expression(0);

    if (current_.type != token_type::right_paren)
      throw_error("expected ')'", current_.begin);

    next_token();
    return node;
  }

  throw_error("expected primary expression", current_.begin);
}

// Read the next token from the input string.
void expression_parser::next_token()
{
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
    return;
  }

  const char current_char = input_[position_];

  // Identifier: [a-zA-Z_][a-zA-Z0-9_]*
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
    return;
  }

  // Integer: [0-9]+
  if (std::isdigit(static_cast<unsigned char>(current_char)) != 0)
  {
    const std::size_t start = position_++;
    while (position_ < input_.size() &&
           std::isdigit(static_cast<unsigned char>(input_[position_])) != 0)
    {
      ++position_;
    }

    current_.type = token_type::integer;
    current_.begin = start;
    current_.length = position_ - start;
    return;
  }

  // Two-character operators.
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
        return;
      }
      break;

    case '>':
      if (next_char == '=')
      {
        current_.type = token_type::greater_equal;
        current_.begin = position_;
        current_.length = 2;
        position_ += 2;
        return;
      }
      break;

    case '=':
      if (next_char == '=')
      {
        current_.type = token_type::equal_equal;
        current_.begin = position_;
        current_.length = 2;
        position_ += 2;
        return;
      }
      break;

    case '!':
      if (next_char == '=')
      {
        current_.type = token_type::not_equal;
        current_.begin = position_;
        current_.length = 2;
        position_ += 2;
        return;
      }
      break;

    case '&':
      if (next_char == '&')
      {
        current_.type = token_type::logical_and;
        current_.begin = position_;
        current_.length = 2;
        position_ += 2;
        return;
      }
      break;

    case '|':
      if (next_char == '|')
      {
        current_.type = token_type::logical_or;
        current_.begin = position_;
        current_.length = 2;
        position_ += 2;
        return;
      }
      break;
    }
  }

  // One-character tokens.
  ++position_;

  switch (current_char)
  {
  case '(':
    current_.type = token_type::left_paren;
    current_.length = 1;
    return;

  case ')':
    current_.type = token_type::right_paren;
    current_.length = 1;
    return;

  case '+':
    current_.type = token_type::plus;
    current_.length = 1;
    return;

  case '-':
    current_.type = token_type::minus;
    current_.length = 1;
    return;

  case '*':
    current_.type = token_type::star;
    current_.length = 1;
    return;

  case '/':
    current_.type = token_type::slash;
    current_.length = 1;
    return;

  case '<':
    current_.type = token_type::less;
    current_.length = 1;
    return;

  case '>':
    current_.type = token_type::greater;
    current_.length = 1;
    return;

  case '!':
    current_.type = token_type::logical_not;
    current_.length = 1;
    return;

  default:
    throw_error("unexpected character", current_.begin);
  }
}

// Return precedence for a supported binary operator.
int expression_parser::binary_precedence(token_type type) const
{
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

// Return the original text slice for a token.
std::string_view expression_parser::token_text(const token &tok) const
{
  return input_.substr(tok.begin, tok.length);
}

// Allocate one AST node from the internal node pool.
const expression_node *expression_parser::make_node(
  std::string_view value,
  const expression_node *left,
  const expression_node *right)
{
  nodes_.emplace_back(value);
  expression_node &node = nodes_.back();
  node.left = left;
  node.right = right;
  return &node;
}

// Throw a parse error with source position.
[[noreturn]] void
expression_parser::throw_error(const char *message, std::size_t position) const
{
  throw std::runtime_error(
    std::string(message) + " at position " + std::to_string(position));
}
