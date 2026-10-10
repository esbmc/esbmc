#include <llm/goto/render.h>

#include <goto-programs/goto_structure.h>
#include <irep2/irep2_utils.h>
#include <util/arith/mp_arith.h>
#include <util/config/config.h>
#include <util/irep/migrate.h>
#include <util/lang/c_expr2string.h>
#include <util/lang/c_qualifiers.h>

#include <algorithm>
#include <cctype>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace llm
{
namespace
{
using id_namest = std::unordered_map<irep_idt, std::string, irep_id_hash>;
using instructiont = goto_programt::instructiont;
using targett = goto_programt::const_targett;

std::string integer_name(unsigned width, bool is_signed)
{
  const auto &c = config.ansi_c;
  std::string base;
  if (width == c.char_width)
    return is_signed ? "signed char" : "unsigned char";
  if (width == c.short_int_width)
    base = "short";
  else if (width == c.int_width)
    base = "int";
  else if (width == c.long_int_width)
    base = "long";
  else if (width == c.long_long_int_width)
    base = "long long";
  else
    base = "_BitInt(" + std::to_string(width) + ")";
  return is_signed ? base : "unsigned " + base;
}

/// `tag-struct S` -> `struct S`.
std::string tag_name(const irep_idt &id)
{
  std::string name = id2string(id);
  if (name.compare(0, 4, "tag-") == 0)
    name.erase(0, 4);
  return name;
}

std::optional<std::string> base_type(const type2tc &type)
{
  switch (type->type_id)
  {
  case type2t::bool_id:
    return "_Bool";
  case type2t::empty_id:
    return "void";
  case type2t::signedbv_id:
  case type2t::unsignedbv_id:
    return integer_name(type->get_width(), is_signedbv_type(type));
  case type2t::floatbv_id:
    return type->get_width() == 32   ? "float"
           : type->get_width() == 64 ? "double"
                                     : "long double";
  case type2t::struct_id:
    return tag_name(to_struct_type(type).name);
  case type2t::union_id:
    return tag_name(to_union_type(type).name);
  case type2t::symbol_id:
    return tag_name(to_symbol_type(type).symbol_name);
  default:
    return std::nullopt;
  }
}

class scoped_expr2stringt : public c_expr2stringt
{
public:
  scoped_expr2stringt(const namespacet &ns, const id_namest &names)
    : c_expr2stringt(ns, 0), names(names)
  {
  }

  std::string declaration(const type2tc &type, const std::string &name)
  {
    if (is_array_type(type))
    {
      const array_type2t &array = to_array_type(type);
      const std::string size = array.size_is_infinite
                                 ? ""
                                 : convert(migrate_expr_back(array.array_size));
      return declaration(array.subtype, name + "[" + size + "]");
    }
    if (is_pointer_type(type))
    {
      const type2tc &target = to_pointer_type(type).subtype;
      const bool wrap = is_array_type(target) || is_code_type(target);
      return declaration(target, wrap ? "(*" + name + ")" : "*" + name);
    }
    if (is_code_type(type))
    {
      const code_type2t &code = to_code_type(type);
      std::string params;
      for (const type2tc &arg : code.arguments)
        params += (params.empty() ? "" : ", ") + declaration(arg, "");
      if (code.ellipsis)
        params += params.empty() ? "..." : ", ...";
      return declaration(code.ret_type, name + "(" + params + ")");
    }
    const std::optional<std::string> base = base_type(type);
    if (!base)
      return convert_rec(migrate_type_back(type), c_qualifierst(), name);
    if (name.empty())
      return *base;
    return *base + " " + name;
  }

protected:
  /// Suffixed so that constants parse back to their type.
  std::string convert_constant(const exprt &src, unsigned &precedence) override
  {
    std::string text = c_expr2stringt::convert_constant(src, precedence);
    const typet &type = src.type();
    if (
      (type.id() != "signedbv" && type.id() != "unsignedbv") || text.empty() ||
      text.find_first_not_of("-0123456789") != std::string::npos)
      return text;
    const unsigned width = string2integer(type.width().as_string()).to_uint64();
    const bool is_unsigned = type.id() == "unsignedbv";
    const auto &c = config.ansi_c;
    std::string suffix;
    if (width == c.int_width)
      suffix = is_unsigned ? "u" : "";
    else if (width == c.long_int_width)
      suffix = is_unsigned ? "ul" : "l";
    else if (width == c.long_long_int_width)
      suffix = is_unsigned ? "ull" : "ll";
    else
      return text;
    // MIN has no literal of its own type.
    const BigInt min = -BigInt::power2(width - 1);
    if (!is_unsigned && string2integer(text) == min)
      return "(-" + integer2string(-(min + 1)) + suffix + " - 1" + suffix + ")";
    return text + suffix;
  }

  std::string convert_symbol(const exprt &src, unsigned &precedence) override
  {
    auto it = names.find(src.identifier());
    if (it != names.end())
      return it->second;
    return c_expr2stringt::convert_symbol(src, precedence);
  }

private:
  const id_namest &names;
};

id_namest names_of(const scopet &scope)
{
  id_namest names;
  for (const auto &[name, symbol] : scope)
    names.emplace(to_symbol2t(symbol).thename, name);
  return names;
}

std::string name_of(const id_namest &names, const irep_idt &id)
{
  auto it = names.find(id);
  return it != names.end() ? it->second : id2string(id);
}

std::string
print_expr(const expr2tc &expr, const id_namest &names, const namespacet &ns)
{
  return scoped_expr2stringt(ns, names).convert(migrate_expr_back(expr));
}

std::string short_name(const irep_idt &id, const namespacet &ns)
{
  const symbolt *symbol = ns.lookup(id);
  std::string name = id2string(symbol ? symbol->name : id);
  std::string::size_type at = name.rfind('@');
  if (at != std::string::npos)
    name.erase(0, at + 1);
  for (char &c : name)
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
      c = '_';
  if (name.empty() || std::isdigit(static_cast<unsigned char>(name[0])))
    name.insert(0, "_");
  return name;
}

void collect_symbols(
  const expr2tc &expr,
  std::vector<expr2tc> &out,
  std::set<irep_idt> &seen)
{
  if (is_nil_expr(expr))
    return;
  if (is_symbol2t(expr))
  {
    const irep_idt &id = to_symbol2t(expr).thename;
    if (!is_code_type(expr) && id != "NULL" && seen.insert(id).second)
      out.push_back(expr);
    return;
  }
  expr->foreach_operand(
    [&](const expr2tc &op) { collect_symbols(op, out, seen); });
}

/// Clashing names get `_2`, `_3`, ...
scopet build_scope(const goto_functiont &function, const namespacet &ns)
{
  std::vector<expr2tc> symbols;
  std::set<irep_idt> seen;
  const code_type2t &type = to_code_type(function.type);
  for (std::size_t i = 0; i < type.arguments.size(); ++i)
    if (!type.argument_names[i].empty())
      collect_symbols(
        symbol2tc(type.arguments[i], type.argument_names[i]), symbols, seen);
  for (const instructiont &i : function.body.instructions)
  {
    if (i.is_decl())
    {
      const code_decl2t &decl = to_code_decl2t(i.code);
      collect_symbols(symbol2tc(decl.type, decl.value), symbols, seen);
    }
    collect_symbols(i.code, symbols, seen);
    collect_symbols(i.guard, symbols, seen);
    for (const expr2tc &inv : i.get_loop_invariants())
      collect_symbols(inv, symbols, seen);
  }

  const std::set<std::string> reserved = {
    "true",
    "false",
    "NULL",
    "_Bool",
    "bool",
    "char",
    "short",
    "int",
    "long",
    "signed",
    "unsigned"};
  std::set<std::string> taken = reserved;
  for (const expr2tc &s : symbols)
    taken.insert(short_name(to_symbol2t(s).thename, ns));

  scopet scope;
  for (const expr2tc &s : symbols)
  {
    std::string name = short_name(to_symbol2t(s).thename, ns);
    if (scope.count(name) || reserved.count(name))
    {
      const std::string base = name;
      for (unsigned k = 2; taken.count(name); ++k)
        name = base + "_" + std::to_string(k);
      taken.insert(name);
    }
    scope.emplace(name, s);
  }
  return scope;
}

class function_renderert
{
public:
  function_renderert(
    const goto_programt &body,
    const id_namest &names,
    const namespacet &ns)
    : body(body), ns(ns), names(names)
  {
  }

  std::string render()
  {
    print(recover_structure(body), 1);
    std::ostringstream out;
    std::set<const instructiont *> labelled;
    for (const linet &line : lines)
    {
      auto label = jump_labels.find(line.anchor);
      if (label != jump_labels.end() && labelled.insert(line.anchor).second)
        out << indent(line.indent) << label->second << ":\n";
      if (!line.text.empty())
        out << indent(line.indent) << line.text << "\n";
    }
    return out.str();
  }

  const std::map<std::string, targett> &loop_labels() const
  {
    return printed_loops;
  }

private:
  struct linet
  {
    const instructiont *anchor;
    unsigned indent;
    std::string text;
  };

  const goto_programt &body;
  const namespacet &ns;
  const id_namest &names;
  std::map<std::string, targett> printed_loops;
  std::map<const instructiont *, std::string> jump_labels;
  std::vector<linet> lines;
  bool after_return = false;

  static std::string indent(unsigned n)
  {
    return std::string(2 * n, ' ');
  }

  std::string expr(const expr2tc &e) const
  {
    return print_expr(e, names, ns);
  }

  void emit(const instructiont *anchor, unsigned depth, std::string text)
  {
    after_return = anchor && anchor->is_return();
    lines.push_back({anchor, depth, std::move(text)});
  }

  std::string guarded(const expr2tc &guard, const std::string &stmt) const
  {
    return is_true(guard) ? stmt : "if (" + expr(guard) + ") " + stmt;
  }

  std::string jump_label(targett target)
  {
    auto label = jump_labels.emplace(
      &*target, "label_" + std::to_string(jump_labels.size() + 1));
    return label.first->second;
  }

  void print(const std::vector<structured_stmtt> &stmts, unsigned depth)
  {
    for (const structured_stmtt &s : stmts)
      print(s, depth);
  }

  void print(const structured_stmtt &s, unsigned depth)
  {
    switch (s.kind)
    {
    case structured_stmtt::INSTRUCTION:
      emit(s.instruction, depth, instruction(*s.instruction));
      break;
    case structured_stmtt::ANCHOR:
      emit(s.instruction, depth, "");
      break;
    case structured_stmtt::BREAK:
      emit(s.instruction, depth, guarded(s.condition, "break;"));
      break;
    case structured_stmtt::CONTINUE:
      emit(s.instruction, depth, guarded(s.condition, "continue;"));
      break;
    case structured_stmtt::GOTO:
      print_goto(s, depth);
      break;
    case structured_stmtt::IF:
      print_if(s, depth);
      break;
    default:
      print_loop(s, depth);
    }
  }

  void print_goto(const structured_stmtt &s, unsigned depth)
  {
    if (
      s.targets.size() == 1 && s.targets.front()->is_end_function() &&
      is_true(s.condition))
    {
      const bool silent = after_return && !s.instruction->is_target();
      emit(s.instruction, depth, silent ? "" : "return;");
      return;
    }
    std::string labels;
    for (targett target : s.targets)
      labels += (labels.empty() ? "" : ", ") + jump_label(target);
    if (s.targets.size() != 1)
      labels = "{" + labels + "}";
    emit(s.instruction, depth, guarded(s.condition, "goto " + labels + ";"));
  }

  void print_if(const structured_stmtt &s, unsigned depth)
  {
    emit(s.instruction, depth, "if (" + expr(s.condition) + ")");
    print_block(s.body, depth);
    if (s.otherwise.empty())
      return;
    emit(nullptr, depth, "else");
    print_block(s.otherwise, depth);
  }

  void print_block(const std::vector<structured_stmtt> &stmts, unsigned depth)
  {
    emit(nullptr, depth, "{");
    print(stmts, depth + 1);
    emit(nullptr, depth, "}");
  }

  void print_loop(const structured_stmtt &s, unsigned depth)
  {
    const std::string label = "L" + std::to_string(s.loop);
    printed_loops.emplace(label, s.head);
    const std::string written = modifies(s.head, s.latch);
    if (!written.empty())
      emit(
        s.instruction, depth, "/* " + label + " modifies: " + written + " */");
    const std::string tag = "/* " + label + " */ ";
    if (s.kind == structured_stmtt::WHILE)
      emit(s.instruction, depth, tag + "while (" + expr(s.condition) + ")");
    else
      emit(
        nullptr,
        depth,
        tag + (s.kind == structured_stmtt::DO_WHILE ? "do" : "while (1)"));
    print_block(s.body, depth);
    if (s.kind == structured_stmtt::DO_WHILE)
      lines.back().text = "} while (" + expr(s.condition) + ");";
  }

  static expr2tc written_object(expr2tc lhs)
  {
    while (is_index2t(lhs) || is_member2t(lhs))
    {
      if (is_member2t(lhs))
        lhs = to_member2t(lhs).source_value;
      else if (is_pointer_type(to_index2t(lhs).source_value))
        return dereference2tc(lhs->type, to_index2t(lhs).source_value);
      else
        lhs = to_index2t(lhs).source_value;
    }
    return lhs;
  }

  /// Direct writes only; callees are not followed.
  std::string modifies(targett head, targett latch) const
  {
    std::set<irep_idt> local;
    std::vector<expr2tc> written;
    for (targett it = head; it != std::next(latch); ++it)
    {
      if (it->is_decl())
        local.insert(to_code_decl2t(it->code).value);
      else if (it->is_assign())
        written.push_back(to_code_assign2t(it->code).target);
      else if (
        it->is_function_call() &&
        !is_nil_expr(to_code_function_call2t(it->code).ret))
        written.push_back(to_code_function_call2t(it->code).ret);
    }
    std::vector<std::string> out;
    for (const expr2tc &lhs : written)
    {
      const expr2tc object = written_object(lhs);
      if (is_symbol2t(object) && local.count(to_symbol2t(object).thename))
        continue;
      const std::string text = expr(object);
      if (std::find(out.begin(), out.end(), text) == out.end())
        out.push_back(text);
    }
    std::string joined;
    for (const std::string &s : out)
      joined += (joined.empty() ? "" : ", ") + s;
    return joined;
  }

  std::string call(const code_function_call2t &call) const
  {
    std::string text = is_nil_expr(call.ret) ? "" : expr(call.ret) + " = ";
    text += expr(call.function) + "(";
    for (std::size_t i = 0; i < call.operands.size(); ++i)
      text += (i ? ", " : "") + expr(call.operands[i]);
    return text + ");";
  }

  std::string assign(const code_assign2t &assign) const
  {
    return expr(assign.target) + " = " + expr(assign.source) + ";";
  }

  std::string declare(const code_decl2t &decl) const
  {
    std::string text = scoped_expr2stringt(ns, names).declaration(
      decl.type, name_of(names, decl.value));
    if (!is_nil_expr(decl.init))
      text += " = " + expr(decl.init);
    return text + ";";
  }

  std::string return_value(const code_return2t &ret) const
  {
    return is_nil_expr(ret.operand) ? "return;"
                                    : "return " + expr(ret.operand) + ";";
  }

  std::string invariants(const instructiont &i) const
  {
    std::string text;
    for (const expr2tc &inv : i.get_loop_invariants())
      text += (text.empty() ? "" : " ") + ("invariant(" + expr(inv) + ");");
    std::string targets;
    for (const expr2tc &target : i.get_loop_assigns_targets())
      targets += (targets.empty() ? "" : ", ") + expr(target);
    if (!targets.empty())
      text += (text.empty() ? "" : " ") + ("assigns(" + targets + ");");
    return text;
  }

  std::string statement(const expr2tc &code) const
  {
    std::string text = expr(code);
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.back())))
      text.pop_back();
    return text;
  }

  std::string instruction(const instructiont &i) const
  {
    switch (i.type)
    {
    case ASSIGN:
      return assign(to_code_assign2t(i.code));
    case DECL:
      return declare(to_code_decl2t(i.code));
    case ASSUME:
      return "assume(" + expr(i.guard) + ");";
    case ASSERT:
      return "assert(" + expr(i.guard) + ");";
    case FUNCTION_CALL:
      return call(to_code_function_call2t(i.code));
    case RETURN:
      return return_value(to_code_return2t(i.code));
    case LOOP_INVARIANT:
      return invariants(i);
    case ATOMIC_BEGIN:
      return "atomic_begin();";
    case ATOMIC_END:
      return "atomic_end();";
    case THROW:
      return "throw;";
    case CATCH:
      return "catch;";
    case OTHER:
      return statement(i.code);
    default:
      return "";
    }
  }
};
} // namespace

rendered_functiont render_function(
  const irep_idt &name,
  const goto_functiont &function,
  const namespacet &ns)
{
  rendered_functiont out;
  out.scope = build_scope(function, ns);
  const id_namest names = names_of(out.scope);

  const code_type2t &type = to_code_type(function.type);
  scoped_expr2stringt printer(ns, names);
  std::string header =
    printer.declaration(type.ret_type, short_name(name, ns)) + "(";
  for (std::size_t i = 0; i < type.arguments.size(); ++i)
    header += (i ? ", " : "") +
              printer.declaration(
                type.arguments[i], name_of(names, type.argument_names[i]));
  header += type.ellipsis ? (type.arguments.empty() ? "..." : ", ...") : "";

  function_renderert renderer(function.body, names, ns);
  out.text = header + ")\n{\n" + renderer.render() + "}\n";
  out.loops = renderer.loop_labels();
  return out;
}

std::string
render_expr(const expr2tc &expr, const scopet &scope, const namespacet &ns)
{
  return print_expr(expr, names_of(scope), ns);
}
} // namespace llm
