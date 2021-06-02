#ifndef SOLIDITY_AST_FRONTEND_SOLIDITY_CONVERT_H_
#define SOLIDITY_AST_FRONTEND_SOLIDITY_CONVERT_H_

//#define __STDC_LIMIT_MACROS
//#define __STDC_FORMAT_MACROS

#include <util/context.h>
#include <util/namespace.h>
#include <util/std_types.h>
#include <nlohmann/json.hpp>

class solidity_convertert
{
public:
  solidity_convertert(
    contextt &_context,
    nlohmann::json &_ast_json);
  virtual ~solidity_convertert() = default;

  bool convert();

protected:
  contextt &context;
  nlohmann::json &ast_json;

  bool convert_top_level_decl();
};

#endif /* SOLIDITY_AST_FRONTEND_SOLIDITY_CONVERT_H_ */
