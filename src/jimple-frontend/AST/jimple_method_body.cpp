#include <jimple-frontend/AST/jimple_method_body.h>
#include <irep2/irep2_expr.h>
#include <jimple-frontend/AST/jimple_declaration.h>
#include <jimple-frontend/AST/jimple_statement.h>
#include <memory>
#include <util/irep/std_code.h>
#include <util/expr/expr_util.h>

// The IREP2 twin of to_exprt below (K.2 of
// docs/roadmap/scope-jimple-irep2.md). migrate_expr's block arm additionally
// splices a child whose legacy statement is "decl-block"; this frontend never
// builds one -- its statements are decl, assign, block, function_call, goto,
// label, return and skip -- so there is nothing to reproduce here.
expr2tc jimple_full_method_body::to_code2t(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  std::vector<expr2tc> ops;
  ops.reserve(this->members.size());

  for (auto const &stmt : this->members)
  {
    auto l = jimple_ast::get_location(class_name, function_name);
    if (stmt->line_location != -1)
      l.set_line(stmt->line_location);

    ops.push_back(stmt->to_code2t(ctx, class_name, function_name, l));
  }

  // migrate_expr reads the legacy block's location through the *const*
  // accessor, so an unlocated block migrates with a nil location, not the
  // default-constructed empty one -- the nil-vs-empty distinction of #6176.
  // A default here renders as a blank location where the round-trip renders
  // "no location". This frontend never locates the block itself.
  // Both locations must be nil, not default-constructed: migrate_expr reads
  // them off the legacy block through the *const* accessors, so an unlocated
  // block migrates nil where a default renders blank -- the nil-vs-empty
  // distinction of #6176. end_location is the one that shows: goto_convert
  // gives END_FUNCTION that location. This frontend locates neither.
  const locationt &nil = static_cast<const locationt &>(get_nil_irep());
  return code_block2tc(ops, nil, nil);
}

exprt jimple_full_method_body::to_exprt(
  contextt &ctx,
  const std::string &class_name,
  const std::string &function_name) const
{
  /* This is a function body, so we create a `code_blockt` and
     * populate it with all its statements (`this->members`) */
  code_blockt block;

  // For each Jimple Statement
  for (auto const &stmt : this->members)
  {
    // Generate the equivalent exprt of the jimple statement
    auto expression = stmt->to_exprt(ctx, class_name, function_name);

    // Get a location for the class and this function
    auto l = jimple_ast::get_location(class_name, function_name);

    // If the original line is known, then we set it
    if (stmt->line_location != -1)
      l.set_line(stmt->line_location);

    expression.location() = l;

    // Add the expression into the block
    block.operands().push_back(expression);
  }

  return block;
}

void jimple_full_method_body::from_json(const json &stmts)
{
  /* In Jimple, locations are set through attributes and it
     * applied to every instruction after it:
     *
     *  \* 2  \*  <--- Comment
     *  a = 3;
     *  b = 4;
     *
     * This means that both statements came from line 2.
     * To solve this, we threat the location as a Statement.
     */
  int inner_location = -1;
  for (const json &stmt : stmts)
  {
    std::shared_ptr<jimple_method_field> to_add;

    auto mode = stmt.at("object").get<std::string>();
    // I think that this is the best way without
    // adding some crazy function pointer.
    auto it = from_map.find(mode);
    if (it == from_map.end())
    {
      log_error("Unknown type {}", stmt.dump(2));
      abort();
    }
    switch (it->second)
    {
    case statement::Declaration:
    {
      jimple_declaration d;
      stmt.get_to(d);
      to_add = std::make_shared<jimple_declaration>(d);
      break;
    }
    case statement::Location:
    {
      /*
         * After parsing the Jimple, the JSON will parse
         * the location as a string and set it to the
         * inner location (see comment above about inner_location)
         */
      std::string location_str;
      stmt.at("line").get_to(location_str);
      inner_location = std::stoi(location_str);
      // Location is not a real statement, continue to the next.
      continue;
    }
    case statement::Identity:
    {
      jimple_identity s;
      stmt.at("statement").at("identity").get_to(s);
      to_add = std::make_shared<jimple_identity>(s);
      break;
    }
    case statement::StaticInvoke:
    {
      jimple_invoke s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_invoke>(s);
      break;
    }
    case statement::VirtualInvoke:
    {
      jimple_invoke s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_invoke>(s);
      break;
    }
    case statement::SpecialInvoke:
    {
      jimple_invoke s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_invoke>(s);
      break;
    }
    case statement::Return:
    {
      jimple_return s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_return>(s);
      break;
    }
    case statement::Label:
    {
      jimple_label s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_label>(s);
      break;
    }
    case statement::Goto:
    {
      jimple_goto s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_goto>(s);
      break;
    }
    case statement::Assignment:
    {
      jimple_assignment s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_assignment>(s);
      break;
    }
    case statement::Throw:
    {
      jimple_throw s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_throw>(s);
      break;
    }
    case statement::If:
    {
      jimple_if s;
      stmt.get_to(s);
      to_add = std::make_shared<jimple_if>(s);
      break;
    }
    default:
      log_error(
        "unsupported jimple statement id {} for key '{}'",
        static_cast<std::underlying_type_t<statement>>(it->second),
        it->first);
      abort();
    }

    to_add->line_location = inner_location;
    members.push_back(std::move(to_add));
  }
}
std::string jimple_full_method_body::to_string() const
{
  std::ostringstream oss;
  for (auto &x : members)
  {
    oss << "\n\t\t" << x->to_string();
  }
  return oss.str();
}
