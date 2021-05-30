// Copyright (c) 2019 Chao Peng
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef SIF_LIBSIF_ASTNODES_H_
#define SIF_LIBSIF_ASTNODES_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <libUtils/Utils.hpp>

namespace Sif{

const std::string TokenSourceUnit = "SourceUnit";
const std::string TokenPragmaDirective = "PragmaDirective";
const std::string TokenImportDirective = "ImportDirective";
const std::string TokenContractDefinition = "ContractDefinition";
const std::string TokenInheritanceSpecifier = "InheritanceSpecifier";
const std::string TokenUsingForDirective = "UsingForDirective";
const std::string TokenStructDefinition = "StructDefinition";
const std::string TokenEnumDefinition = "EnumDefinition";
const std::string TokenEnumValue = "EnumValue";
const std::string TokenParameterList = "ParameterList";
const std::string TokenFunctionDefinition = "FunctionDefinition";
const std::string TokenVariableDeclaration = "VariableDeclaration";
const std::string TokenModifierDefinition = "ModifierDefinition";
const std::string TokenModifierInvocation = "ModifierInvocation";
const std::string TokenEventDefinition = "EventDefinition";
const std::string TokenElementaryTypeName = "ElementaryTypeName";
const std::string TokenUserDefinedTypeName = "UserDefinedTypeName";
const std::string TokenFunctionTypeName = "FunctionTypeName";
const std::string TokenMapping = "Mapping";
const std::string TokenArrayTypeName = "ArrayTypeName";
const std::string TokenInlineAssembly = "InlineAssembly";
const std::string TokenBlock = "Block";
const std::string TokenPlaceholderStatement = "PlaceholderStatement";
const std::string TokenIfStatement = "IfStatement";
const std::string TokenDoWhileStatement = "DoWhileStatement";
const std::string TokenWhileStatement = "WhileStatement";
const std::string TokenForStatement = "ForStatement";
const std::string TokenContinue = "Continue";
const std::string TokenBreak = "Break";
const std::string TokenReturn = "Return";
const std::string TokenThrow = "Throw";
const std::string TokenEmitStatement = "EmitStatement";
const std::string TokenVariableDeclarationStatement = "VariableDeclarationStatement";
const std::string TokenExpressionStatement = "ExpressionStatement";
const std::string TokenConditional = "Conditional";
const std::string TokenAssignment = "Assignment";
const std::string TokenTupleExpression = "TupleExpression";
const std::string TokenUnaryOperation = "UnaryOperation";
const std::string TokenBinaryOperation = "BinaryOperation";
const std::string TokenFunctionCall = "FunctionCall";
const std::string TokenNewExpression = "NewExpression";
const std::string TokenMemberAccess = "MemberAccess";
const std::string TokenIndexAccess = "IndexAccess";
const std::string TokenIdentifier = "Identifier";
const std::string TokenElementaryTypeNameExpression = "ElementaryTypeNameExpression";
const std::string TokenLiteral = "Literal,";
const std::string TokenLiterals = "Literals";
const std::string TokenSource = "Source:";
const std::string TokenType = "Type:";

const std::list<std::string> TokenList{
    TokenSourceUnit,
    TokenPragmaDirective,
    TokenImportDirective,
    TokenContractDefinition,
    TokenInheritanceSpecifier,
    TokenUsingForDirective,
    TokenStructDefinition,
    TokenEnumDefinition,
    TokenEnumValue,
    TokenParameterList,
    TokenFunctionDefinition,
    TokenVariableDeclaration,
    TokenModifierDefinition,
    TokenModifierInvocation,
    TokenEventDefinition,
    TokenElementaryTypeName,
    TokenUserDefinedTypeName,
    TokenFunctionTypeName,
    TokenMapping,
    TokenArrayTypeName,
    TokenInlineAssembly,
    TokenBlock,
    TokenPlaceholderStatement,
    TokenIfStatement,
    TokenDoWhileStatement,
    TokenWhileStatement,
    TokenForStatement,
    TokenContinue,
    TokenBreak,
    TokenReturn,
    TokenThrow,
    TokenEmitStatement,
    TokenVariableDeclarationStatement,
    TokenExpressionStatement,
    TokenConditional,
    TokenAssignment,
    TokenTupleExpression,
    TokenUnaryOperation,
    TokenBinaryOperation,
    TokenFunctionCall,
    TokenNewExpression,
    TokenMemberAccess,
    TokenIndexAccess,
    TokenIdentifier,
    TokenElementaryTypeNameExpression,
    TokenLiteral,
    TokenLiterals,
    //TokenSource,
    //TokenType
};

const std::list<std::string> ExpressionTokenList{
    TokenConditional,
    TokenAssignment,
    TokenTupleExpression,
    TokenUnaryOperation,
    TokenBinaryOperation,
    TokenFunctionCall,
    TokenNewExpression,
    TokenMemberAccess,
    TokenIndexAccess,
    TokenIdentifier,
    TokenElementaryTypeNameExpression,
    TokenLiteral
};

const std::list<std::string> StatementTokenList{
    TokenBlock,
    TokenPlaceholderStatement,
    TokenIfStatement,
    TokenDoWhileStatement,
    TokenWhileStatement,
    TokenForStatement,
    TokenContinue,
    TokenBreak,
    TokenReturn,
    TokenThrow,
    TokenEmitStatement,
    TokenVariableDeclarationStatement,
    TokenExpressionStatement,
    TokenInlineAssembly
};

enum NodeType {
    NodeTypeSourceUnit,
    NodeTypeRoot,
    NodeTypePragmaDirective,
    NodeTypeImportDirective,
    NodeTypeUsingForDirective,
    NodeTypeVariableDeclaration,
    NodeTypeStructDefinition,
    NodeTypeParameterList,
    NodeTypeEventDefinition,
    NodeTypeBlockNode,
    NodeTypeFunctionDefinition,
    NodeTypeContractDefinition,
    NodeTypeFunctionCall,
    NodeTypeEnumDefinition,
    NodeTypeEnumValue,
    NodeTypeModifierDefinition,
    NodeTypeModifierInvocation,
    NodeTypeMapping,
    NodeTypeInlineAssembly,
    NodeTypePlaceholderStatement,
    NodeTypeIfStatement,
    NodeTypeDoWhileStatement,
    NodeTypeWhileStatement,
    NodeTypeForStatement,
    NodeTypeContinue,
    NodeTypeBreak,
    NodeTypeReturn,
    NodeTypeThrow,
    NodeTypeEmitStatement,
    NodeTypeVariableDeclarationStatement,
    NodeTypeExpressionStatement,
    NodeTypeConditional,
    NodeTypeAssignment,
    NodeTypeTupleExpression,
    NodeTypeUnaryOperation,
    NodeTypeBinaryOperation,
    NodeTypeNewExpression,
    NodeTypeMemberAccess,
    NodeTypeIndexAccess,
    NodeTypeIdentifier,
    NodeTypeElementaryTypeNameExpression,
    NodeTypeLiteral,
    NodeTypeElementaryTypeName,
    NodeTypeUserDefinedTypeName,
    NodeTypeFunctionTypeName,
    NodeTypeArrayTypeName
};

typedef std::vector<std::string> Literals;

class Indentation {
public:
    Indentation() : tab_width(4), use_spaces(true), current_tab_width(0) {}
    explicit Indentation(const int& _tab_width) : tab_width(_tab_width), use_spaces(true), current_tab_width(0) {}
    Indentation(const int& _tab_width, const bool& _use_spaces) : tab_width(_tab_width), use_spaces(_use_spaces), current_tab_width(0) {}
    Indentation(const int& _tab_width, const bool& _use_spaces, const int& _current_tab_width) : tab_width(_tab_width), use_spaces(_use_spaces), current_tab_width(_current_tab_width) {}
    Indentation(const Indentation& _indentation);
    std::string str() const;
    Indentation& operator++();
    Indentation operator++(int);
    Indentation& operator--();
    Indentation operator--(int);
    Indentation& operator=(const Indentation& _indentation );
private:
    int tab_width;
    bool use_spaces;
    int current_tab_width;
};
typedef std::shared_ptr<Indentation> IndentationPtr;

std::ostream& operator<<(std::ostream& _os, const Indentation& _indentation);
std::string operator+(const std::string& _str, const Indentation& _indentation);
std::string operator+(const Indentation& _indentation, const std::string& _str);

class ASTNode;
typedef std::shared_ptr<ASTNode> ASTNodePtr;

class ASTNode {
public:
    explicit ASTNode(NodeType _node_type) : node_type(_node_type), text_before(""), text_after("") {}
    virtual std::string source_code(Indentation& _indentation) = 0;
    NodeType get_node_type() const;
    size_t size();
    void insert_text_before(const std::string& _text);
    void insert_text_after(const std::string& _text);
    std::string get_added_text_before() const;
    std::string get_added_text_after() const;
protected:
    void append_sub_node(const ASTNodePtr& _node);
    void delete_sub_node(const unsigned int& x);
    void update_sub_node(const unsigned int& x, const ASTNodePtr _node);
    ASTNodePtr get_sub_node(const unsigned int& x) const;
    ASTNodePtr operator[] (const unsigned int& x);

    NodeType node_type;
    std::vector<ASTNodePtr> ast_nodes;
    std::string text_before;
    std::string text_after;
};

class RootNode : public ASTNode {
public:
    RootNode() : ASTNode(NodeTypeRoot), import(""), pragma("") {}
    std::string source_code(Indentation& _indentation);
    void set_import(const std::string& _import);
    std::string get_import() const;
    void set_pragma(const std::string& _pragma);
    std::string get_pragma() const;

    void add_field(const ASTNodePtr& _node);
    void delete_field(const unsigned int& x);
    void update_field(const unsigned int& x, const ASTNodePtr& _node);
    ASTNodePtr get_field(const unsigned int& x);
    size_t num_fields();
    ASTNodePtr operator[] (const unsigned int& x);
private:
    std::string import;
    std::string pragma;
};
typedef std::shared_ptr<RootNode> RootNodePtr;

class PragmaDirectiveNode : public ASTNode {
public:
    PragmaDirectiveNode() : ASTNode(NodeTypePragmaDirective) {}
    std::string source_code(Indentation& _indentation);
    void set_literals(const Literals& _literals);
    Literals get_literals();
private:
    Literals literals;
};
typedef std::shared_ptr<PragmaDirectiveNode> PragmaDirectiveNodePtr;

class ImportDirectiveNode : public ASTNode {
public:
    ImportDirectiveNode() : ASTNode(NodeTypeImportDirective), original(""), file(""), symbol_aliases(""), unit_alias("") {}
    explicit ImportDirectiveNode(const std::string& _original) : ASTNode(NodeTypeImportDirective), original(_original) {}
    void set_file(const std::string& _file);
    void set_symbol_aliases(const std::string& _symbol_aliases);
    void set_unit_alias(const std::string& _unit_aliases);
    void set_original(const std::string& _original);
    std::string get_file();
    std::string get_symbol_aliases();
    std::string get_unit_aliases();
    std::string get_original();
    std::string source_code(Indentation& _indentation);
private:
    std::string file;
    std::string symbol_aliases;
    std::string unit_alias;
    std::string original;
};
typedef std::shared_ptr<ImportDirectiveNode> ImportDirectiveNodePtr;

class UsingForDirectiveNode : public ASTNode {
public:
    UsingForDirectiveNode() : ASTNode(NodeTypeUsingForDirective), A(""), B("") {}
    UsingForDirectiveNode(std::string& _A, std::string& _B) : ASTNode(NodeTypeUsingForDirective), A(_A), B(_B) {}
    std::string source_code(Indentation& _indentation);
    void set_using(const std::string& _using);
    void set_for(const std::string& _for);
    std::string get_using();
    std::string get_for();
private:
    std::string A;
    std::string B;
};
typedef std::shared_ptr<UsingForDirectiveNode> UsingForDirectiveNodePtr;

class VariableDeclarationNode : public ASTNode {
public:
    VariableDeclarationNode() : ASTNode(NodeTypeVariableDeclaration), type(nullptr), initial_value(nullptr) {}
    VariableDeclarationNode(const ASTNodePtr& _type, std::string& _variable_name) : ASTNode(NodeTypeVariableDeclaration), type(_type), variable_name(_variable_name), initial_value(nullptr) {}
    std::string source_code(Indentation& _indentation);
    void set_type(const ASTNodePtr& _type);
    void set_variable_name(const std::string& _variable_name);
    void set_initial_value(const ASTNodePtr& _initial_value);
    ASTNodePtr get_type();
    std::string get_variable_name();
    ASTNodePtr get_initial_value();
private:
    ASTNodePtr type;
    std::string variable_name;
    ASTNodePtr initial_value;
};
typedef std::shared_ptr<VariableDeclarationNode> VariableDeclarationNodePtr;

class VariableDeclarationStatementNode : public ASTNode {
public:
    VariableDeclarationStatementNode() : ASTNode(NodeTypeVariableDeclarationStatement), decl(nullptr), value(nullptr) {}
    VariableDeclarationStatementNode(const VariableDeclarationNodePtr& _decl, const ASTNodePtr& _value) : ASTNode(NodeTypeVariableDeclarationStatement), decl(_decl), value(_value) {}
    std::string source_code(Indentation& _indentation);
    VariableDeclarationNodePtr get_decl() const;
    ASTNodePtr get_value() const;
    void set_decl(const VariableDeclarationNodePtr& _decl);
    void set_value(const ASTNodePtr& _value);
private:
    VariableDeclarationNodePtr decl;
    ASTNodePtr value;
};
typedef std::shared_ptr<VariableDeclarationStatementNode> VariableDeclarationStatementNodePtr;

class IdentifierNode : public ASTNode {
public:
    explicit IdentifierNode(const std::string& _name) : ASTNode(NodeTypeIdentifier), name(_name) {}
    std::string source_code(Indentation& _indentation);
    std::string get_name() const;
    void set_name(const std::string& _name);
private:
    std::string name;
};
typedef std::shared_ptr<IdentifierNode> IdentifierNodePtr;

class StructDefinitionNode : public ASTNode {
public:
    explicit StructDefinitionNode(const std::string& _name) : ASTNode(NodeTypeStructDefinition), name(_name) {};
    std::string source_code(Indentation& _indentation);
    std::string get_name() const;
    void set_name(const std::string& _name);
    void add_field(const ASTNodePtr& _node);
    void delete_field(const unsigned int& x);
    void update_field(const unsigned int& x, const ASTNodePtr& _node);
    ASTNodePtr get_field(const unsigned int& x);
    size_t num_fields();
    ASTNodePtr operator[] (const unsigned int& x);
private:
    std::string name;
};
typedef std::shared_ptr<StructDefinitionNode> StructDefinitionNodePtr;

class ParameterListNode : public ASTNode {
public:
    ParameterListNode() : ASTNode(NodeTypeParameterList) {};
    std::string source_code(Indentation& _indentation);
    void add_parameter(const ASTNodePtr& _node);
    void delete_parameter(const unsigned int& x);
    void update_parameter(const unsigned int& x, const ASTNodePtr& _node);
    ASTNodePtr get_parameter(const unsigned int& x);
    size_t num_parameters();
    ASTNodePtr operator[] (const unsigned int& x);
};
typedef std::shared_ptr<ParameterListNode> ParameterListNodePtr;

class EventDefinitionNode : public ASTNode {
public:
    explicit EventDefinitionNode(const std::string& _name) : ASTNode(NodeTypeEventDefinition), name(_name) {}
    std::string source_code(Indentation& _indentation);
    std::string get_name() const;
    void set_name(const std::string& _name);
    void set_argument_list(const ParameterListNodePtr& _node);
    ParameterListNodePtr get_argument_list() const;
private:
    std::string name;
    ParameterListNodePtr argument_list;
};
typedef std::shared_ptr<EventDefinitionNode> EventDefinitionNodePtr;

class ExpressionStatementNode : public ASTNode {
public:
    ExpressionStatementNode() : ASTNode(NodeTypeExpressionStatement) {}
    std::string source_code(Indentation& _indentation);
    ASTNodePtr get_expression() const;
    void set_expression(const ASTNodePtr& _expression);
private:
    ASTNodePtr expression;
};
typedef std::shared_ptr<ExpressionStatementNode> ExpressionStatementNodePtr;

class EmitStatementNode : public ASTNode {
public:
    EmitStatementNode() : ASTNode(NodeTypeEmitStatement) {}
    std::string source_code(Indentation& _indentation);
    ASTNodePtr get_event_call() const;
    void set_event_call(const ASTNodePtr& _event_call);
private:
    ASTNodePtr event_call;
};
typedef std::shared_ptr<EmitStatementNode> EmitStatementNodePtr;

class IndexAccessNode : public ASTNode {
public:
    IndexAccessNode() : ASTNode(NodeTypeIndexAccess) {}
    std::string source_code(Indentation& _indentation);
    ASTNodePtr get_identifier() const;
    ASTNodePtr get_index_value() const;
    void set_identifier(const ASTNodePtr& _identifier);
    void set_index_value(const ASTNodePtr& _index_value);
private:
    ASTNodePtr identifier;
    ASTNodePtr index_value;
};
typedef std::shared_ptr<IndexAccessNode> IndexAccessNodePtr;

class BinaryOperationNode : public ASTNode {
public:
    BinaryOperationNode(std::string& _op, ASTNodePtr _left, ASTNodePtr _right) : ASTNode(NodeTypeBinaryOperation), op(_op), left_hand_operand(_left), right_hand_operand(_right) {}
    std::string source_code(Indentation& _indentation);
    std::string get_operator() const;
    void set_operator(const std::string& _operator);
    ASTNodePtr get_left_hand_operand() const;
    ASTNodePtr get_right_hand_operand() const;
    void set_left_hand_operand(const ASTNodePtr& _operand);
    void set_right_hand_operand(const ASTNodePtr& _operand);
    std::string get_return_type_str() const;
    void set_return_type_str(const std::string& _return_type_str);
private:
    std::string op;
    std::string return_type_str;
    ASTNodePtr left_hand_operand;
    ASTNodePtr right_hand_operand;
};
typedef std::shared_ptr<BinaryOperationNode> BinaryOperationNodePtr;

class UnaryOperationNode : public ASTNode {
public:
    UnaryOperationNode(std::string& _op, ASTNodePtr _operand, bool _is_prefix) : ASTNode(NodeTypeUnaryOperation), op(_op), operand(_operand), is_prefix(_is_prefix) {}
    std::string source_code(Indentation& _indentation);
    std::string get_operator() const;
    void set_operator(const std::string& _operator);
    ASTNodePtr get_operand() const;
    void set_operand(const ASTNodePtr& _operand);
    bool operation_is_prefix() const;
    void operation_is_prefix(bool _boolean);
private:
    std::string op;
    ASTNodePtr operand;
    bool is_prefix;
};
typedef std::shared_ptr<UnaryOperationNode> UnaryOperationNodePtr;

class LiteralNode : public ASTNode {
public:
    explicit LiteralNode(const std::string& _literal) : ASTNode(NodeTypeLiteral), literal(_literal) {}
    std::string source_code(Indentation& _indentation);
    void set_literal(const std::string& _literal);
    std::string get_literal() const;
private:
    std::string literal;
};
typedef std::shared_ptr<LiteralNode> LiteralNodePtr;

class TupleExpressionNode : public ASTNode {
public:
    TupleExpressionNode() : ASTNode(NodeTypeTupleExpression) {}
    std::string source_code(Indentation& _indentation);
    void add_member(const ASTNodePtr& _node);
    void delete_member(const unsigned int& x);
    void update_member(const unsigned int& x, const ASTNodePtr& _node);
    ASTNodePtr get_member(const unsigned int& x);
    size_t num_members();
    ASTNodePtr operator[] (const unsigned int& x);
private:
};
typedef std::shared_ptr<TupleExpressionNode> TupleExpressionNodePtr;

class BlockNode : public ASTNode {
public:
    BlockNode() : ASTNode(NodeTypeBlockNode) {}
    std::string source_code(Indentation& _indentation);
    void add_statement(const ASTNodePtr& _node);
    void delete_statement(const unsigned int& x);
    void update_statement(const unsigned int& x, const ASTNodePtr& _node);
    ASTNodePtr get_statement(const unsigned int& x);
    size_t num_statements();
    ASTNodePtr operator[] (const unsigned int& x);
private:
};
typedef std::shared_ptr<BlockNode> BlockNodePtr;

class ReturnNode : public ASTNode {
public:
    ReturnNode() : ASTNode(NodeTypeReturn) {}
    explicit ReturnNode(ASTNodePtr _operand) : ASTNode(NodeTypeReturn), operand(_operand) {}
    std::string source_code(Indentation& _indentation);
    ASTNodePtr get_operand() const;
    void set_operand(const ASTNodePtr& _operand);
private:
    ASTNodePtr operand;
};
typedef std::shared_ptr<ReturnNode> ReturnNodePtr;

class ModifierDefinitionNode : public ASTNode {
public:
    ModifierDefinitionNode() : ASTNode(NodeTypeModifierDefinition), params(nullptr) {}
    ModifierDefinitionNode(std::string& _name, ParameterListNodePtr _params, ASTNodePtr _body) : ASTNode(NodeTypeModifierDefinition), name(_name), params(_params), body(_body) {}
    std::string source_code(Indentation& _indentation);
    std::string get_name() const;
    ParameterListNodePtr get_params() const;
    ASTNodePtr get_body() const;
    void set_name(const std::string& _name);
    void set_params(const ParameterListNodePtr& _params);
    void set_body(const ASTNodePtr& _body);
private:
    std::string name;
    ParameterListNodePtr params;
    ASTNodePtr body;
};
typedef std::shared_ptr<ModifierDefinitionNode> ModifierDefinitionNodePtr;

class ModifierInvocationNode : public ASTNode {
public:
    ModifierInvocationNode() : ASTNode(NodeTypeModifierInvocation) {}
    explicit ModifierInvocationNode(const std::string& _name) : ASTNode(NodeTypeModifierInvocation), name(_name) {}
    std::string source_code(Indentation& _indentation);
    void add_argument(const ASTNodePtr& _node);
    void delete_argument(const unsigned int& x);
    void update_argument(const unsigned int& x, const ASTNodePtr& _node);
    ASTNodePtr get_argument(const unsigned int& x);
    size_t num_arguments();
    ASTNodePtr operator[] (const unsigned int& x);
    std::string get_name() const;
    void set_name(const std::string& _name);
private:
    std::string name;
};
typedef std::shared_ptr<ModifierInvocationNode> ModifierInvocationNodePtr;

class FunctionDefinitionNode : public ASTNode {
public:
    FunctionDefinitionNode() : ASTNode(NodeTypeFunctionDefinition), is_constructor(false) {};
    FunctionDefinitionNode(std::string& _name, std::string& _qualifier, ParameterListNodePtr _params, ParameterListNodePtr _returns, BlockNodePtr _function_body) : ASTNode(NodeTypeFunctionDefinition), name(_name), qualifier(_qualifier), params(_params), returns(_returns), function_body(_function_body), is_constructor(false) {}
    std::string source_code(Indentation& _indentation);
    void add_modifier_invocation(const ModifierInvocationNodePtr& _node);
    void delete_modifier_invocation(const unsigned int& x);
    void update_modifier_invocation(const unsigned int& x, const ModifierInvocationNodePtr& _node);
    ModifierInvocationNodePtr get_modifier_invocation(const unsigned int& x);
    size_t num_modifier_invocations();
    void set_name(const std::string& _name);
    void set_qualifier(const std::string& _qualifier);
    void set_params(const ParameterListNodePtr& _params);
    void set_returns(const ParameterListNodePtr& _returns);
    void set_function_body(const BlockNodePtr& _function_body);
    void set_is_constructor(const bool& _is_constructor);
    std::string get_name() const;
    std::string get_qualifier() const;
    ParameterListNodePtr get_params() const;
    ParameterListNodePtr get_returns() const;
    BlockNodePtr get_function_body() const;
    bool function_is_constructor() const;
private:
    std::string name;
    std::string qualifier;
    ParameterListNodePtr params;
    ParameterListNodePtr returns;
    std::vector<ModifierInvocationNodePtr> modifier_invocation;
    BlockNodePtr function_body;
    bool is_constructor;
};
typedef std::shared_ptr<FunctionDefinitionNode> FunctionDefinitionNodePtr;

class FunctionCallNode : public ASTNode {
public:
    explicit FunctionCallNode(ASTNodePtr _callee) : ASTNode(NodeTypeFunctionCall), callee(_callee) {}
    std::string source_code(Indentation& _indentation);
    void add_argument(const ASTNodePtr& _node);
    void delete_argument(const unsigned int& x);
    void update_argument(const unsigned int& x, const ASTNodePtr& _node);
    ASTNodePtr get_argument(const unsigned int& x);
    size_t num_arguments();
    void set_callee(const ASTNodePtr& _callee);
    ASTNodePtr get_callee() const;
private:
    ASTNodePtr callee;
    // arguments is stored in subnodes
};
typedef std::shared_ptr<FunctionCallNode> FunctionCallNodePtr;

class MemberAccessNode : public ASTNode {
public:
    MemberAccessNode(ASTNodePtr _identifier, std::string& _member) : ASTNode(NodeTypeMemberAccess), identifier(_identifier), member(_member) {}
    std::string source_code(Indentation& _indentation);
    void set_identifier(const ASTNodePtr& _identifier);
    ASTNodePtr get_identifier() const;
    void set_member(const std::string& _member);
    std::string get_member() const;
private:
    ASTNodePtr identifier;
    std::string member;
};
typedef std::shared_ptr<MemberAccessNode> MemberAccessNodePtr;

class ElementaryTypeNameExpressionNode : public ASTNode{
public:
    explicit ElementaryTypeNameExpressionNode(std::string& _name) : ASTNode(NodeTypeElementaryTypeNameExpression), name(_name) {}
    std::string source_code(Indentation& _indentation);
    std::string get_name() const;
    void set_name(const std::string& _name);
private:
    std::string name;
};
typedef std::shared_ptr<ElementaryTypeNameExpressionNode> ElementaryTypeNameExpressionNodePtr;

class ContractDefinitionNode : public ASTNode {
public:
    explicit ContractDefinitionNode(std::string& _name) : ASTNode(NodeTypeContractDefinition), name(_name), inherit_from{}, is_library(false) {}
    std::string source_code(Indentation& _indentation);
    void add_inherit_from(const std::string& _inherit_from);
    void delete_inherit_from(const unsigned int& x);
    void update_inherit_from(const unsigned int& x, const std::string& _inherit_from);
    std::string get_inherit_from(const unsigned int& x);
    size_t num_inherit_from() const;
    void set_as_library();
    void set_as_non_library();
    bool contract_is_library() const;
    void set_name(const std::string& _name);
    std::string get_name() const; 
    void add_member(const ASTNodePtr& _node);
    void delete_member(const unsigned int& x);
    void update_member(const unsigned int& x, const ASTNodePtr& _node);
    ASTNodePtr get_member(const unsigned int& x);
    size_t num_members();
    ASTNodePtr operator[] (const unsigned int& x);
private:
    std::string name;
    std::vector<std::string> inherit_from;
    bool is_library;
};
typedef std::shared_ptr<ContractDefinitionNode> ContractDefinitionNodePtr;

class IfStatementNode : public ASTNode {
public:
    IfStatementNode(ASTNodePtr _condition, ASTNodePtr _if_then, ASTNodePtr _if_else) : ASTNode(NodeTypeIfStatement), condition(_condition), if_then(_if_then), if_else(_if_else) {}
    std::string source_code(Indentation& _indentation);
    void set_condition(const ASTNodePtr& _condition);
    ASTNodePtr get_condition() const;
    void set_then(const ASTNodePtr& _then);
    ASTNodePtr get_then() const;
    void set_else(const ASTNodePtr& _else);
    ASTNodePtr get_else() const;
private:
    ASTNodePtr condition;
    ASTNodePtr if_then;
    ASTNodePtr if_else;
};
typedef std::shared_ptr<IfStatementNode> IfStatementNodePtr;

class DoWhileStatementNode : public ASTNode {
public:
    DoWhileStatementNode(ASTNodePtr _condition, ASTNodePtr _loop_body) : ASTNode(NodeTypeDoWhileStatement), condition(_condition), loop_body(_loop_body) {}
    std::string source_code(Indentation& _indentation);
    void set_condition(const ASTNodePtr& _condition);
    ASTNodePtr get_condition() const;
    void set_loop_body(const ASTNodePtr& _loop_body);
    ASTNodePtr get_loop_body() const;
private:
    ASTNodePtr condition;
    ASTNodePtr loop_body;
};
typedef std::shared_ptr<DoWhileStatementNode> DoWhileStatementNodePtr;


class WhileStatementNode : public ASTNode {
public:
    WhileStatementNode(ASTNodePtr _condition, ASTNodePtr _loop_body) : ASTNode(NodeTypeWhileStatement), condition(_condition), loop_body(_loop_body) {}
    std::string source_code(Indentation& _indentation);
    void set_condition(const ASTNodePtr& _condition);
    ASTNodePtr get_condition() const;
    void set_loop_body(const ASTNodePtr& _loop_body);
    ASTNodePtr get_loop_body() const;
private:
    ASTNodePtr condition;
    ASTNodePtr loop_body;
};
typedef std::shared_ptr<WhileStatementNode> WhileStatementNodePtr;


class ForStatementNode : public ASTNode {
public:
    ForStatementNode(ASTNodePtr _init, ASTNodePtr _condition, ASTNodePtr _increment, ASTNodePtr _body) : ASTNode(NodeTypeForStatement), init(_init), condition(_condition), increment(_increment), body(_body) {}
    std::string source_code(Indentation& _indentation);
    ASTNodePtr get_init() const;
    ASTNodePtr get_condition() const;
    ASTNodePtr get_increment() const;
    ASTNodePtr get_body() const;
    void set_init(const ASTNodePtr& _init);
    void set_condition(const ASTNodePtr& _condition);
    void set_increment(const ASTNodePtr& _increment);
    void set_body(const ASTNodePtr& _body);
private:
    ASTNodePtr init;
    ASTNodePtr condition;
    ASTNodePtr increment;
    ASTNodePtr body;
};
typedef std::shared_ptr<ForStatementNode> ForStatementNodePtr;

class ConditionalNode : public ASTNode {
public:
    ConditionalNode(ASTNodePtr _condition, ASTNodePtr _yes, ASTNodePtr _no) : ASTNode(NodeTypeConditional),  condition(_condition), yes(_yes), no(_no) {}
    std::string source_code(Indentation& _indentation);
    ASTNodePtr get_condition() const;
    ASTNodePtr get_yes() const;
    ASTNodePtr get_no() const;
    void set_condition(const ASTNodePtr& _condition);
    void set_yes(const ASTNodePtr& _yes);
    void set_no(const ASTNodePtr& _no);
private:
    ASTNodePtr condition;
    ASTNodePtr yes;
    ASTNodePtr no;
};
typedef std::shared_ptr<ConditionalNode> ConditionalNodePtr;

class AssignmentNode : public ASTNode {
public:
    AssignmentNode() : ASTNode(NodeTypeAssignment) {}
    explicit AssignmentNode(const std::string& _op) : ASTNode(NodeTypeAssignment), op(_op) {}
    std::string source_code(Indentation& _indentation);
    std::string get_operator() const;
    void set_operator(const std::string& _operator);
    ASTNodePtr get_left_hand_operand() const;
    ASTNodePtr get_right_hand_operand() const;
    void set_left_hand_operand(const ASTNodePtr& _operand);
    void set_right_hand_operand(const ASTNodePtr& _operand);
private:
    std::string op;
    ASTNodePtr left_hand_operand;
    ASTNodePtr right_hand_operand;
};
typedef std::shared_ptr<AssignmentNode> AssignmentNodePtr;

class BreakNode : public ASTNode {
public:
    BreakNode() : ASTNode(NodeTypeBreak) {};
    std::string source_code(Indentation& _indentation);
};
typedef std::shared_ptr<BreakNode> BreakNodePtr;

class ContinueNode : public ASTNode {
public:
    ContinueNode() : ASTNode(NodeTypeContinue) {};
    std::string source_code(Indentation& _indentation);
};
typedef std::shared_ptr<ContinueNode> ContinueNodePtr;

class NewExpresionNode : public ASTNode {
public:
    explicit NewExpresionNode(ASTNodePtr _type_name) : ASTNode(NodeTypeNewExpression), type_name(_type_name) {}
    std::string source_code(Indentation& _indentation);
    void set_type_name(const ASTNodePtr& _type_name);
    ASTNodePtr get_type_name() const;
private:
    ASTNodePtr type_name;
};
typedef std::shared_ptr<NewExpresionNode> NewExpresionNodePtr;

class EnumDefinitionNode : public ASTNode {
public:
    explicit EnumDefinitionNode(const std::string& _name) : ASTNode(NodeTypeEnumDefinition), name(_name) {}
    std::string source_code(Indentation& _indentation);
    void add_member(const ASTNodePtr& _node);
    void delete_member(const unsigned int& x);
    void update_member(const unsigned int& x, const ASTNodePtr& _node);
    ASTNodePtr get_member(const unsigned int& x);
    size_t num_members();
    void set_name(const std::string& _name);
    std::string get_name() const;
private:
    std::string name;
};
typedef std::shared_ptr<EnumDefinitionNode> EnumDefinitionNodePtr;

class EnumValueNode : public ASTNode {
public:
    explicit EnumValueNode(const std::string& _name) : ASTNode(NodeTypeEnumValue), name(_name) {}
    std::string source_code(Indentation& _indentation);
    std::string get_name() const;
    void set_name(const std::string& _name);
private:
    std::string name;
};
typedef std::shared_ptr<EnumValueNode> EnumValueNodePtr;

class ThrowNode : public ASTNode {
public:
    ThrowNode() : ASTNode(NodeTypeThrow) {};
    std::string source_code(Indentation& _indentation);
};
typedef std::shared_ptr<ThrowNode> ThrowNodePtr;

class PlaceHolderStatement : public ASTNode {
public:
    explicit PlaceHolderStatement(const std::string& _place_holder) : ASTNode(NodeTypePlaceholderStatement), placeholder(_place_holder) {};
    std::string source_code(Indentation& _indentation);
    std::string get_placeholder() const;
    void set_placeholder(const std::string& _placeholder);
private:
    std::string placeholder;
};
typedef std::shared_ptr<PlaceHolderStatement> PlaceHolderStatementPtr;

class MappingNode : public ASTNode {
public:
    MappingNode(ASTNodePtr _key_type, ASTNodePtr _value_type) : ASTNode(NodeTypeMapping), key_type(_key_type), value_type(_value_type) {}
    std::string source_code(Indentation& _indentation);
    ASTNodePtr get_key_type() const;
    ASTNodePtr get_value_type() const;
    void set_key_type(const ASTNodePtr& _key_type);
    void set_value_type(const ASTNodePtr& _value_type);
private:
    ASTNodePtr key_type;
    ASTNodePtr value_type;
};
typedef std::shared_ptr<MappingNode> MappingNodePtr;

class ElementaryTypeNameNode : public ASTNode {
public:
    explicit ElementaryTypeNameNode(const std::string& _type_name) : ASTNode(NodeTypeElementaryTypeName), type_name(_type_name) {}
    std::string source_code(Indentation& _indentation);
    void set_type_name(const std::string& _type_name);
    std::string get_type_name() const;
private:
    std::string type_name;
};
typedef std::shared_ptr<ElementaryTypeNameNode> ElementaryTypeNameNodePtr;

class UserDefinedTypeNameNode : public ASTNode {
public:
    explicit UserDefinedTypeNameNode(const std::string& _type_name) : ASTNode(NodeTypeUserDefinedTypeName), type_name(_type_name) {}
    std::string source_code(Indentation& _indentation);
    void set_type_name(const std::string& _type_name);
    std::string get_type_name() const;
private:
    std::string type_name;
};
typedef std::shared_ptr<UserDefinedTypeNameNode> UserDefinedTypeNameNodePtr;

class FunctionTypeNameNode : public ASTNode {
public:
    FunctionTypeNameNode(ASTNodePtr _params, ASTNodePtr _returns) : ASTNode(NodeTypeFunctionTypeName), params(_params), returns(_returns) {}
    std::string source_code(Indentation& _indentation);
    ASTNodePtr get_params() const;
    ASTNodePtr get_returns() const;
    void set_params(const ASTNodePtr& _params);
    void set_returns(const ASTNodePtr& _returns);
private:
    ASTNodePtr params;
    ASTNodePtr returns;
};
typedef std::shared_ptr<FunctionTypeNameNode> FunctionTypeNameNodePtr;

class ArrayTypeNameNode : public ASTNode {
public:
    ArrayTypeNameNode(ASTNodePtr _base_type, ASTNodePtr _size) : ASTNode(NodeTypeArrayTypeName), base_type(_base_type), size(_size) {}
    std::string source_code(Indentation& _indentation);
    ASTNodePtr get_base_type() const;
    ASTNodePtr get_size() const;
    void set_base_type(const ASTNodePtr& _base_type);
    void set_size(const ASTNodePtr& _size);
private:
    ASTNodePtr base_type;
    ASTNodePtr size;
};
typedef std::shared_ptr<ArrayTypeNameNode> ArrayTypeNameNodePtr;

class InlineAssemblyNode : public ASTNode {
public:
    InlineAssemblyNode() : ASTNode(NodeTypeInlineAssembly) {};
    std::string source_code(Indentation& _indentation);
    void set_source(std::string& _source);
    std::string get_source();
private:
    std::string source;
};
typedef std::shared_ptr<InlineAssemblyNode> InlineAssemblyNodePtr;

}

#endif //SIF_LIBSIF_ASTNODES_H_
