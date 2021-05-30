// Copyright (c) 2019 Chao Peng
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef SIF_LIBSIF_ASTANALYSER_H_
#define SIF_LIBSIF_ASTANALYSER_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include <libUtils/Utils.hpp>
#include "ASTNodes.hpp"

namespace Sif{

class ASTAnalyser {
public:
    ASTAnalyser(std::stringstream& _ast_sstream, nlohmann::json& _jsonast, const bool& single_file, const std::string& file_name, const std::string& _visitor_arg);
    RootNodePtr analyse();
    void set_do_not_produce_source(bool _do_not_produce_source = true);

private:
    std::list<std::string> ast_lines;
    std::list<std::string>::iterator ptr_ast_line;
    nlohmann::json ast_json;
    std::vector<ContractDefinitionNodePtr> contracts;
    ContractDefinitionNodePtr current_contract;
    std::string current_contract_name;
    unsigned int num_functions_current_contract;
    std::string visitor_arg;
    bool do_not_produce_source;

    void get_next_token(const std::string& token);
    std::string get_next_token();
    void remove_escapes(std::string& _str);
    VariableDeclarationNodePtr handle_variable_declaration();
    ParameterListNodePtr handle_parameter_list();
    BlockNodePtr handle_block();
    ExpressionStatementNodePtr handle_expression_statament();
    BinaryOperationNodePtr handle_binary_operation();
    UnaryOperationNodePtr handle_unary_operation();
    LiteralNodePtr handle_literal();
    TupleExpressionNodePtr handle_tuple_expression();
    VariableDeclarationStatementNodePtr handle_variable_declaration_statament();
    IdentifierNodePtr handle_identifier();
    ReturnNodePtr handle_return();
    FunctionCallNodePtr handle_function_call();
    MemberAccessNodePtr handle_member_access();
    EmitStatementNodePtr handle_emit_statement();
    IndexAccessNodePtr handle_index_access();
    ElementaryTypeNameExpressionNodePtr handle_elementary_type_name_expression();
    ConditionalNodePtr handle_conditional();
    AssignmentNodePtr handle_assignment();
    IfStatementNodePtr handle_if_statement();
    ForStatementNodePtr handle_for_statement();
    DoWhileStatementNodePtr handle_do_while_statament();
    WhileStatementNodePtr handle_while_statement();
    ASTNodePtr get_value_equivalent_node(std::string& token);
    ASTNodePtr get_statement_equivalent_node(std::string& token);
    ASTNodePtr get_unknown(std::string& token);
    NewExpresionNodePtr handle_new_expression();
    EnumDefinitionNodePtr handle_enum_definition();
    EnumValueNodePtr handle_enum_value();
    PlaceHolderStatementPtr handle_placeholder();
    ModifierInvocationNodePtr handle_modifier_invocation();
    ASTNodePtr get_type_name(std::string& token);
    int get_current_indentation();

    std::string get_function_qualifier(const std::string& contract_name, const std::string& _function_name, const std::string& _type);
    bool function_is_constructor(const std::string& _contract_name, const std::string& _function_name);
};

}

#endif //SIF_LIBSIF_ASTANALYSER_H_