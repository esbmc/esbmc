// Copyright (c) 2019 chao
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "ASTNodes.hpp"
#include "ASTVisitor.hpp"

namespace Sif{

Indentation::Indentation(const Indentation& _indentation) {
    tab_width = _indentation.tab_width;
    use_spaces = _indentation.use_spaces;
    current_tab_width = _indentation.current_tab_width;
}

Indentation& Indentation::operator++() {
    current_tab_width += tab_width;
    return *this;
}

Indentation Indentation::operator++(int) {
    Indentation result(*this);
    ++(*this);
    return result;
}

Indentation& Indentation::operator--() {
    current_tab_width -= tab_width;
    if (current_tab_width < 0) current_tab_width = 0;
    return *this;
}

Indentation Indentation::operator--(int) {
    Indentation result(*this);
    --(*this);
    return result;
}

Indentation& Indentation::operator=(const Indentation& _indentation ) {
    tab_width = _indentation.tab_width;
    use_spaces = _indentation.use_spaces;
    current_tab_width = _indentation.current_tab_width;
    return *this;
}

std::string Indentation::str() const {
    std::string result;
    for (int i = 0; i < current_tab_width; ++i) {
        result = result + (use_spaces? ' ': '\t');
    }
    return result;
}

std::ostream& operator<<(std::ostream& _os, const Indentation& _indentation) {
    _os << _indentation.str();
    return _os;
}

std::string operator+(const std::string& _str, const Indentation& _indentation) {
    std::string result = _str + _indentation.str();
    return result;
}

std::string operator+(const Indentation& _indentation, const std::string& _str) {
    std::string result = _indentation.str() + _str;
    return result;
}

NodeType ASTNode::get_node_type() const {
    return node_type;
}

void ASTNode::insert_text_before(const std::string& _text) {
    text_before = _text;
}

void ASTNode::insert_text_after(const std::string& _text) {
    text_after = _text;
}

std::string ASTNode::get_added_text_before() const {
    return text_before;
}

std::string ASTNode::get_added_text_after() const {
    return text_after;
}

void ASTNode::append_sub_node(const ASTNodePtr& _node) {
    ast_nodes.push_back(_node);
}

void ASTNode::delete_sub_node(const unsigned int& x) {
    ast_nodes.erase(ast_nodes.begin() + x);
}

void ASTNode::update_sub_node(const unsigned int& x, const ASTNodePtr _node) {
    ast_nodes[x] = _node;
}

ASTNodePtr ASTNode::get_sub_node(const unsigned int& x) const {
    return ast_nodes[x];
}

size_t ASTNode::size() {
    return ast_nodes.size();
}

ASTNodePtr ASTNode::operator[] (const unsigned int& x) {
    return ast_nodes[x];
}

std::string RootNode::source_code(Indentation& _indentation) {
    visit(this);
    std::stringstream result;
    result << text_before << pragma << "\n" << import << "\n";
    for (auto it = ast_nodes.begin(); it != ast_nodes.end(); ++it) {
        result << (*it)->source_code(_indentation);
    }
    result << text_after;
    return result.str();
}

void RootNode::set_import(const std::string& _import) {
    import = _import;
}

std::string RootNode::get_import() const{
    return import;
}

void RootNode::set_pragma(const std::string& _pragma) {
    pragma = _pragma;
}

std::string RootNode::get_pragma() const{
    return pragma;
}

void RootNode::add_field(const ASTNodePtr& _node) {
    append_sub_node(_node);
}

size_t RootNode::num_fields() {
    return size();
}

ASTNodePtr RootNode::operator[] (const unsigned int& x) {
    return get_sub_node(x);
}

void RootNode::update_field(const unsigned int& x, const ASTNodePtr& _node) {
    update_sub_node(x, _node);
}

void RootNode::delete_field(const unsigned int& x) {
    delete_sub_node(x);
}

ASTNodePtr RootNode::get_field(const unsigned int& x) {
    return get_sub_node(x);
}

std::string PragmaDirectiveNode::source_code(Indentation& _indentation) {
    visit(this);
    std::string result = text_before + _indentation + "pragma " + literals[0] + " " + literals[1] + literals[2] + literals[3] + ";" + text_after;
    return result;
}

void PragmaDirectiveNode::set_literals(const Literals& _literals) {
    literals = _literals;
}

Literals PragmaDirectiveNode::get_literals() {
    return literals;
}

void ImportDirectiveNode::set_file(const std::string& _file) {
    file = _file;
}

void ImportDirectiveNode::set_symbol_aliases(const std::string& _symbol_aliases) {
    symbol_aliases = _symbol_aliases;
}

void ImportDirectiveNode::set_unit_alias(const std::string& _unit_aliases) {
    unit_alias = _unit_aliases;
}

void ImportDirectiveNode::set_original(const std::string& _original) {
    original = _original;
}

std::string ImportDirectiveNode::get_file() {
    return file;
}

std::string ImportDirectiveNode::get_symbol_aliases() {
    return symbol_aliases;
}

std::string ImportDirectiveNode::get_unit_aliases() {
    return unit_alias;
}

std::string ImportDirectiveNode::get_original() {
    return original;
}

std::string ImportDirectiveNode::source_code(Indentation& _indentation) {
    visit(this);
    if (original != "") {
        return text_before + _indentation + original + text_after;
    }
    std::string result = text_before;
    if (unit_alias == "") {
        result = _indentation+ "import \"" + file + "\";";
    } else {
        if (symbol_aliases == "") {
            result = _indentation + "import * as " + unit_alias + " from \"" + file + "\"";
        } else {
            result = _indentation + "import " + symbol_aliases + " as " + unit_alias + " from \"" + file + "\"";
        }
    }
    result = result + text_after;
    return result;
}

std::string UsingForDirectiveNode::source_code(Indentation& _indentation) {
    visit(this);
    std::string result = text_before + _indentation + "using " + A + " for " + B + ";" + text_after;
    return result;
}

void UsingForDirectiveNode::set_using(const std::string& _using) {
    A = _using;
}

void UsingForDirectiveNode::set_for(const std::string& _for) {
    B = _for;
}

std::string UsingForDirectiveNode::get_using() {
    return A;
}

std::string UsingForDirectiveNode::get_for() {
    return B;
}

std::string VariableDeclarationNode::source_code(Indentation& _indentation) {
    visit(this);
    std::string result = text_before;
    Indentation empty_indentation(0);
    if (initial_value != nullptr) {
        result = _indentation + type->source_code(empty_indentation) + " " + variable_name + " = " + initial_value->source_code(empty_indentation);
    } else {
        result = _indentation + type->source_code(empty_indentation) + " " + variable_name;
    }
    result = result + text_after;
    return result;
}

void VariableDeclarationNode::set_type(const ASTNodePtr& _type) {
    type = _type;
}

void VariableDeclarationNode::set_variable_name(const std::string& _variable_name) {
    variable_name = _variable_name;
}

void VariableDeclarationNode::set_initial_value(const ASTNodePtr& _initial_value) {
    initial_value = _initial_value;
}

ASTNodePtr VariableDeclarationNode::get_type() {
    return type;
}

std::string VariableDeclarationNode::get_variable_name() {
    return variable_name;
}

ASTNodePtr VariableDeclarationNode::get_initial_value() {
    return initial_value;
}

std::string VariableDeclarationStatementNode::source_code(Indentation& _indentation) {
    visit(this);
    std::string result = text_before;
    Indentation empty_indentation(0);
    if (value == nullptr) {
        result = result + _indentation + decl->source_code(empty_indentation) + ";";
    } else {
        result = result + _indentation + decl->source_code(empty_indentation) + " = " + value->source_code(empty_indentation) + ";";
    }
    result = result + text_after;
    return result;
}

VariableDeclarationNodePtr VariableDeclarationStatementNode::get_decl() const {
    return decl;
}

ASTNodePtr VariableDeclarationStatementNode::get_value() const {
    return value;
}

void VariableDeclarationStatementNode::set_decl(const VariableDeclarationNodePtr& _decl) {
    decl = _decl;
}

void VariableDeclarationStatementNode::set_value(const ASTNodePtr& _value) {
    value = _value;
}

std::string IdentifierNode::source_code(Indentation& _indentation) {
    visit(this);
    return text_before + _indentation + name + text_after;
}

std::string IdentifierNode::get_name() const {
    return name;
}

void IdentifierNode::set_name(const std::string& _name) {
    name = _name;
}

std::string StructDefinitionNode::source_code(Indentation& _indentation) {
    visit(this);
    std::string result = text_before + _indentation +  "struct " + name + " {\n";
    ++_indentation;
    for (auto it = ast_nodes.begin(); it != ast_nodes.end(); ++it) {
        result = result + (*it)->source_code(_indentation) + ";\n";
    }
    --_indentation;
    result = result + _indentation + "}\n" + text_after;
    return result;
}

std::string StructDefinitionNode::get_name() const {
    return name;
}

void StructDefinitionNode::set_name(const std::string& _name) {
    name = _name;
}

void StructDefinitionNode::add_field(const ASTNodePtr& _node) {
    append_sub_node(_node);
}

size_t StructDefinitionNode::num_fields() {
    return size();
}

ASTNodePtr StructDefinitionNode::operator[] (const unsigned int& x) {
    return get_sub_node(x);
}

void StructDefinitionNode::update_field(const unsigned int& x, const ASTNodePtr& _node) {
    update_sub_node(x, _node);
}

void StructDefinitionNode::delete_field(const unsigned int& x) {
    delete_sub_node(x);
}

ASTNodePtr StructDefinitionNode::get_field(const unsigned int& x) {
    return get_sub_node(x);
}

std::string ParameterListNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    if (ast_nodes.size() == 0) return text_before + _indentation + "()" + text_after;
    std::string result = text_before + _indentation + "(";
    for (auto it = ast_nodes.begin(); it != ast_nodes.end(); ++it) {
        result = result + (*it)->source_code(empty_indentation) + ", ";
    }
    result = result.substr(0, result.length()-2) + ")" + text_after;
    return result;
}

void ParameterListNode::add_parameter(const ASTNodePtr& _node) {
    append_sub_node(_node);
}

void ParameterListNode::delete_parameter(const unsigned int& x) {
    delete_sub_node(x);
}

void ParameterListNode::update_parameter(const unsigned int& x, const ASTNodePtr& _node) {
    update_sub_node(x, _node);
}

ASTNodePtr ParameterListNode::get_parameter(const unsigned int& x) {
    return get_sub_node(x);
}

size_t ParameterListNode::num_parameters() {
    return size();
}

ASTNodePtr ParameterListNode::operator[] (const unsigned int& x) {
    return get_sub_node(x);
}

void EventDefinitionNode::set_argument_list(const ParameterListNodePtr& _node) {
    argument_list = _node;
}

ParameterListNodePtr EventDefinitionNode::get_argument_list() const {
    return argument_list;
}

std::string EventDefinitionNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    std::string result = text_before + _indentation + "event " + name + argument_list->source_code(empty_indentation) + ";\n" + text_after;
    return result;
}

std::string EventDefinitionNode::get_name() const {
    return name;
}

void EventDefinitionNode::set_name(const std::string& _name) {
    name = _name;
}

std::string ExpressionStatementNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    return text_before + _indentation + expression->source_code(empty_indentation) + ";" + text_after;
}

ASTNodePtr ExpressionStatementNode::get_expression() const {
    return expression;
}

void ExpressionStatementNode::set_expression(const ASTNodePtr& _expression) {
    expression = _expression;
}

std::string EmitStatementNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    return text_before + _indentation + "emit " + event_call -> source_code(empty_indentation) + ";" + text_after;
}

ASTNodePtr EmitStatementNode::get_event_call() const {
    return event_call;
}

void EmitStatementNode::set_event_call(const ASTNodePtr& _event_call) {
    event_call = _event_call;
}

std::string IndexAccessNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    return text_before + _indentation + identifier->source_code(empty_indentation) + "[" + index_value->source_code(empty_indentation) + "]" + text_after;
}

ASTNodePtr IndexAccessNode::get_identifier() const {
    return identifier;
}

ASTNodePtr IndexAccessNode::get_index_value() const {
    return index_value;
}

void IndexAccessNode::set_identifier(const ASTNodePtr& _identifier) {
    identifier = _identifier;
}

void IndexAccessNode::set_index_value(const ASTNodePtr& _index_value) {
    index_value = _index_value;
}

std::string BinaryOperationNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    std::string result = text_before + _indentation
                        + left_hand_operand->source_code(empty_indentation) 
                        + " " + op + " " 
                        + right_hand_operand->source_code(empty_indentation)
                        + text_after;
    return result;
}

std::string BinaryOperationNode::get_return_type_str() const {
    return return_type_str;
}

void BinaryOperationNode::set_return_type_str(const std::string& _return_type_str) {
    return_type_str = _return_type_str;
}

std::string BinaryOperationNode::get_operator() const {
    return op;
}

void BinaryOperationNode::set_operator(const std::string& _operator) {
    op = _operator;
}

ASTNodePtr BinaryOperationNode::get_left_hand_operand() const {
    return left_hand_operand;    
}

ASTNodePtr BinaryOperationNode::get_right_hand_operand() const {
    return right_hand_operand;
}

void BinaryOperationNode::set_left_hand_operand(const ASTNodePtr& _operand) {
    left_hand_operand = _operand;
}

void BinaryOperationNode::set_right_hand_operand(const ASTNodePtr& _operand) {
    right_hand_operand = _operand;
}

std::string UnaryOperationNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    if (is_prefix) {
        if (op == "delete") {
            return text_before + _indentation + op + " " + operand->source_code(empty_indentation) + text_after;
        } else {
            return text_before + _indentation + op + operand->source_code(empty_indentation) + text_after;
        }
    } else {
        return text_before + _indentation + operand->source_code(empty_indentation) + op + text_after;
    }
}

std::string UnaryOperationNode::get_operator() const {
    return op;
}

void UnaryOperationNode::set_operator(const std::string& _operator) {
    op = _operator;
}

ASTNodePtr UnaryOperationNode::get_operand() const {
    return operand;
}

void UnaryOperationNode::set_operand(const ASTNodePtr& _operand) {
    operand = _operand;
}

bool UnaryOperationNode::operation_is_prefix() const {
    return is_prefix;
}

void UnaryOperationNode::operation_is_prefix(bool _boolean) {
    is_prefix = _boolean;
}

std::string LiteralNode::source_code(Indentation& _indentation) {
    visit(this);
    return text_before + _indentation + literal + text_after;
}

void LiteralNode::set_literal(const std::string& _literal) {
    literal = _literal;
}

std::string LiteralNode::get_literal() const {
    return literal;
}

std::string TupleExpressionNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    std::string result = text_before + _indentation + "(";
    for (auto it = ast_nodes.begin(); it != ast_nodes.end(); ++it) {
        result = result + (*it)->source_code(empty_indentation) + ", ";
    }
    result = result.substr(0, result.length()-2) + ")" + text_after;
    return result;
}

void TupleExpressionNode::add_member(const ASTNodePtr& _node) {
    append_sub_node(_node);
}

void TupleExpressionNode::delete_member(const unsigned int& x) {
    delete_sub_node(x);
}

void TupleExpressionNode::update_member(const unsigned int& x, const ASTNodePtr& _node) {
    update_sub_node(x, _node);
}

ASTNodePtr TupleExpressionNode::get_member(const unsigned int& x) {
    return get_sub_node(x);
}

size_t TupleExpressionNode::num_members() {
    return size();
}

ASTNodePtr TupleExpressionNode::operator[] (const unsigned int& x) {
    return get_sub_node(x);
}

std::string BlockNode::source_code(Indentation& _indentation) {
    visit(this);
    std::string result = text_before + "{\n";
    _indentation++;
    for (auto it = ast_nodes.begin(); it != ast_nodes.end(); ++it) {
        result = result + (*it)->source_code(_indentation) + "\n";
    }
    _indentation--;
    result = result + _indentation + "}" + text_after;
    return result;
}

void BlockNode::add_statement(const ASTNodePtr& _node) {
    append_sub_node(_node);
}

void BlockNode::delete_statement(const unsigned int& x) {
    delete_sub_node(x);
}

void BlockNode::update_statement(const unsigned int& x, const ASTNodePtr& _node) {
    update_sub_node(x, _node);
}

ASTNodePtr BlockNode::get_statement(const unsigned int& x) {
    return get_sub_node(x);
}

size_t BlockNode::num_statements() {
    return size();
}

ASTNodePtr BlockNode::operator[] (const unsigned int& x) {
    return get_sub_node(x);
}

std::string ReturnNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    std::string result = text_before + _indentation + "return";
    if (operand) {
        result = result + " " + operand->source_code(empty_indentation) + ";";
    } else {
        result = result + ";";
    }
    result = result + text_after;
    return result;
}

ASTNodePtr ReturnNode::get_operand() const {
    return operand;
}

void ReturnNode::set_operand(const ASTNodePtr& _operand) {
    operand = _operand;
}

std::string ModifierDefinitionNode::source_code(Indentation& _indentation) {
    visit(this);
    std::string params_str;
    Indentation empty_indentation(0);
    if (params == nullptr) params_str = "()";
    else params_str = params->source_code(empty_indentation);
    std::string result = text_before + _indentation + "modifier " + name + params_str + " ";
    result = result + body->source_code(_indentation) + text_after;
    return result;
}

std::string ModifierDefinitionNode::get_name() const {
    return name;
}

ParameterListNodePtr ModifierDefinitionNode::get_params() const {
    return params;
}

ASTNodePtr ModifierDefinitionNode::get_body() const {
    return body;
}

void ModifierDefinitionNode::set_name(const std::string& _name) {
    name = _name;
}

void ModifierDefinitionNode::set_params(const ParameterListNodePtr& _params) {
    params = _params;
}

void ModifierDefinitionNode::set_body(const ASTNodePtr& _body) {
    body = _body;
}

std::string ModifierInvocationNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    if (ast_nodes.size() == 0) {
        return text_before + _indentation + name + "()" + text_after;
    } else {
        std::string result = text_before + _indentation + name + "(";
        for (auto it = ast_nodes.begin(); it != ast_nodes.end(); ++it) {
            result = result + (*it)->source_code(empty_indentation) + ", ";
        }
        result = result.substr(0, result.length()-2) + ")" + text_after;
        return result;
    }
}

std::string ModifierInvocationNode::get_name() const {
    return name;
}

void ModifierInvocationNode::set_name(const std::string& _name) {
    name = _name;
}

void ModifierInvocationNode::add_argument(const ASTNodePtr& _node) {
    append_sub_node(_node);
}

void ModifierInvocationNode::delete_argument(const unsigned int& x) {
    delete_sub_node(x);
}

void ModifierInvocationNode::update_argument(const unsigned int& x, const ASTNodePtr& _node) {
    update_sub_node(x, _node);
}

ASTNodePtr ModifierInvocationNode::get_argument(const unsigned int& x) {
    return get_sub_node(x);
}

size_t ModifierInvocationNode::num_arguments() {
    return size();
}

ASTNodePtr ModifierInvocationNode::operator[] (const unsigned int& x) {
    return get_sub_node(x);
}

std::string FunctionDefinitionNode::source_code(Indentation& _indentation) {
    visit(this);
    std::string result = text_before + _indentation + "";
    Indentation empty_indentation(0);
    if (name == "") {
        if (is_constructor) {
            result = result + "constructor" + params->source_code(empty_indentation);
        } else {
            result = result + "function " + params->source_code(empty_indentation);
        }
    } else {
        result = result + "function " + name + params->source_code(empty_indentation);
    }
    if (qualifier != "") result = result + " " + qualifier;
    if (modifier_invocation.size() != 0) {
        for (auto it = modifier_invocation.begin(); it != modifier_invocation.end(); ++it) {
            result = result + " " + (*it)->source_code(empty_indentation);
        }
    }
    if (returns->num_parameters() != 0) result = result + " returns" + returns->source_code(empty_indentation);
    if (function_body == nullptr) {
        result = result + ";";
    } else {
        result = result + " " + function_body->source_code(_indentation) + "\n";
    }
    result = result + text_after;
    return result;
}

void FunctionDefinitionNode::add_modifier_invocation(const ModifierInvocationNodePtr& _node) {
    modifier_invocation.push_back(_node);
}

void FunctionDefinitionNode::delete_modifier_invocation(const unsigned int& x) {
    modifier_invocation.erase(modifier_invocation.begin() + x);
}

void FunctionDefinitionNode::update_modifier_invocation(const unsigned int& x, const ModifierInvocationNodePtr& _node) {
    modifier_invocation[x] = _node;
}

ModifierInvocationNodePtr FunctionDefinitionNode::get_modifier_invocation(const unsigned int& x) {
    return modifier_invocation[x];
}

size_t FunctionDefinitionNode::num_modifier_invocations() {
    return modifier_invocation.size();
}

void FunctionDefinitionNode::set_name(const std::string& _name) {
    name = _name;
}

void FunctionDefinitionNode::set_qualifier(const std::string& _qualifier) {
    qualifier = _qualifier;
}

void FunctionDefinitionNode::set_params(const ParameterListNodePtr& _params) {
    params = _params;
}

void FunctionDefinitionNode::set_returns(const ParameterListNodePtr& _returns) {
    returns = _returns;
}

void FunctionDefinitionNode::set_function_body(const BlockNodePtr& _function_body) {
    function_body = _function_body;
}

void FunctionDefinitionNode::set_is_constructor(const bool& _is_constructor) {
    is_constructor = _is_constructor;
}

std::string FunctionDefinitionNode::get_name() const {
    return name;
}

std::string FunctionDefinitionNode::get_qualifier() const {
    return qualifier;
}

ParameterListNodePtr FunctionDefinitionNode::get_params() const {
    return params;
}

ParameterListNodePtr FunctionDefinitionNode::get_returns() const {
    return returns;
}

BlockNodePtr FunctionDefinitionNode::get_function_body() const {
    return function_body;
}

bool FunctionDefinitionNode::function_is_constructor() const {
    return is_constructor;
}

std::string FunctionCallNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    if (ast_nodes.size()) {
        std::string result = text_before + _indentation + callee->source_code(empty_indentation) + "(";
        for (auto it = ast_nodes.begin(); it != ast_nodes.end(); ++it) {
            result = result + (*it)->source_code(empty_indentation) + ", ";
        }
        result = result.substr(0, result.length()-2) + ")" + text_after;
        return result;
    } else {
        return text_before + callee->source_code(empty_indentation) + "()" + text_after;
    }
}

void FunctionCallNode::add_argument(const ASTNodePtr& _node) {
    append_sub_node(_node);
}

void FunctionCallNode::delete_argument(const unsigned int& x) {
    delete_sub_node(x);
}

void FunctionCallNode::update_argument(const unsigned int& x, const ASTNodePtr& _node) {
    update_sub_node(x, _node);
}

ASTNodePtr FunctionCallNode::get_argument(const unsigned int& x) {
    return get_sub_node(x);
}

size_t FunctionCallNode::num_arguments() {
    return size();
}

void FunctionCallNode::set_callee(const ASTNodePtr& _callee) {
    callee = _callee;
}

ASTNodePtr FunctionCallNode::get_callee() const {
    return callee;
}

std::string MemberAccessNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    return text_before + identifier->source_code(empty_indentation) + "." + member + text_after;
}

void MemberAccessNode::set_identifier(const ASTNodePtr& _identifier) {
    identifier = _identifier;
}

ASTNodePtr MemberAccessNode::get_identifier() const {
    return identifier;
}

void MemberAccessNode::set_member(const std::string& _member) {
    member = _member;
}

std::string MemberAccessNode::get_member() const {
    return member;
}

std::string ElementaryTypeNameExpressionNode::source_code(Indentation& _indentation) {
    visit(this);
    return text_before + _indentation + name + text_after;
}

std::string ElementaryTypeNameExpressionNode::get_name() const {
    return name;
}

void ElementaryTypeNameExpressionNode::set_name(const std::string& _name) {
    name = _name;
}

std::string ContractDefinitionNode::source_code(Indentation& _indentation) {
    visit(this);
    std::string result = text_before;
    if (is_library) {
        result = _indentation + "library " + name;
    } else {
        result = _indentation + "contract " + name;
    }
    if (inherit_from.size()) {
        auto it = inherit_from.begin();
        result = result + " is " + *it;
        ++it;
        while (it != inherit_from.end()) {
            result = result + ", " + *it;
            ++it;
        }
    }
    result = result + " {\n";
    ++_indentation;
    for (auto it = ast_nodes.begin(); it != ast_nodes.end(); ++it) {
        std::string sub_source_code = (*it)->source_code(_indentation);
        if ((*it)->get_node_type() == NodeTypeVariableDeclaration) sub_source_code = sub_source_code + ";"; // varibale declared in contract-level needs a ';'
        sub_source_code = sub_source_code + "\n";
        result = result + sub_source_code;
    }
    --_indentation;
    result = _indentation + result + "}\n" + text_after;
    return result;
}

void ContractDefinitionNode::add_inherit_from(const std::string& _inherit_from) {
    inherit_from.push_back(_inherit_from);
}

void ContractDefinitionNode::delete_inherit_from(const unsigned int& x) {
    inherit_from.erase(inherit_from.begin() + x);
}

void ContractDefinitionNode::update_inherit_from(const unsigned int& x, const std::string& _inherit_from) {
    inherit_from[x] = _inherit_from;
}

std::string ContractDefinitionNode::get_inherit_from(const unsigned int& x) {
    return inherit_from[x];
}

size_t ContractDefinitionNode::num_inherit_from() const {
    return inherit_from.size();
}

void ContractDefinitionNode::set_as_library() {
    is_library = true;
}

void ContractDefinitionNode::set_as_non_library() {
    is_library = false;
}

bool ContractDefinitionNode::contract_is_library() const {
    return is_library;
}

void ContractDefinitionNode::set_name(const std::string& _name) {
    name = _name;
}

std::string ContractDefinitionNode::get_name() const {
    return name;
}

void ContractDefinitionNode::add_member(const ASTNodePtr& _node) {
    append_sub_node(_node);
}

void ContractDefinitionNode::delete_member(const unsigned int& x) {
    delete_sub_node(x);
}

void ContractDefinitionNode::update_member(const unsigned int& x, const ASTNodePtr& _node) {
    update_sub_node(x, _node);
}

ASTNodePtr ContractDefinitionNode::get_member(const unsigned int& x) {
    return get_sub_node(x);
}

size_t ContractDefinitionNode::num_members() {
    return size();
}

ASTNodePtr ContractDefinitionNode::operator[] (const unsigned int& x) {
    return get_sub_node(x);
}

std::string IfStatementNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    std::string result = text_before + _indentation + "if (" + condition->source_code(empty_indentation) + ") " + if_then->source_code(_indentation);
    if (if_else != nullptr) result = result + " else " + if_else->source_code(_indentation) + "\n";
    else result = result  + "\n";
    result = result + text_after;
    return result; 
}

void IfStatementNode::set_condition(const ASTNodePtr& _condition){
    condition = _condition;
}

ASTNodePtr IfStatementNode::get_condition() const{
    return condition;
}

void IfStatementNode::set_then(const ASTNodePtr& _then){
    if_then = _then;
}

ASTNodePtr IfStatementNode::get_then() const{
    return if_then;
}

void IfStatementNode::set_else(const ASTNodePtr& _else){
    if_else = _else;
}

ASTNodePtr IfStatementNode::get_else() const{
    return if_else;
}

std::string DoWhileStatementNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    std::string result = text_before + _indentation + "do " + loop_body->source_code(_indentation) + " while (" + condition->source_code(empty_indentation) + ");\n" + text_after;
    return result;
}

void DoWhileStatementNode::set_condition(const ASTNodePtr& _condition){
    condition = _condition;
}

ASTNodePtr DoWhileStatementNode::get_condition() const{
    return condition;    
}

void DoWhileStatementNode::set_loop_body(const ASTNodePtr& _loop_body){
    loop_body = _loop_body;
}

ASTNodePtr DoWhileStatementNode::get_loop_body() const{
    return loop_body;
}

std::string WhileStatementNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    std::string result = text_before + _indentation + "while(" + condition->source_code(empty_indentation) + ") " + loop_body->source_code(_indentation) + "\n" + text_after;
    return result;
}

void WhileStatementNode::set_condition(const ASTNodePtr& _condition){
    condition = _condition;
}

ASTNodePtr WhileStatementNode::get_condition() const{
    return condition;    
}

void WhileStatementNode::set_loop_body(const ASTNodePtr& _loop_body){
    loop_body = _loop_body;
}

ASTNodePtr WhileStatementNode::get_loop_body() const{
    return loop_body;
}

std::string ForStatementNode::source_code(Indentation& _indentation) {
    visit(this);
    std::string condition_str = "; ";
    Indentation empty_indentation(0);
    if (condition != nullptr) condition_str = condition->source_code(empty_indentation) + "; ";

    std::string init_str = "; ";
    if (init != nullptr) init_str = init->source_code(empty_indentation) + " ";

    std::string increment_str = "";
    if (increment != nullptr) {
        increment_str = increment->source_code(empty_indentation);
        size_t increment_str_len = increment_str.length();
        for (int i = increment_str.length() - 1; i >= 0; --i) {
            char ch = increment_str[i];
            if (ch == ' ' || ch == '\n') {
                --increment_str_len;
            } else if (ch == ';') {
                --increment_str_len;
                break;
            } else {
                break;
            }
        }
        increment_str = increment_str.substr(0, increment_str_len);
    }

    std::string result = text_before 
                         + _indentation + "for (" + init_str + condition_str + increment_str + ") " 
                         + body->source_code(_indentation) + "\n"
                         + text_after;
    return result;
}

ASTNodePtr ForStatementNode::get_init() const{
    return init;
}

ASTNodePtr ForStatementNode::get_condition() const{
    return condition;
}

ASTNodePtr ForStatementNode::get_increment() const{
    return increment;
}

ASTNodePtr ForStatementNode::get_body() const{
    return body;
}

void ForStatementNode::set_init(const ASTNodePtr& _init){
    init = _init;
}

void ForStatementNode::set_condition(const ASTNodePtr& _condition){
    condition = _condition;
}

void ForStatementNode::set_increment(const ASTNodePtr& _increment){
    increment = _increment;
}

void ForStatementNode::set_body(const ASTNodePtr& _body){
    body = _body;
}

std::string ConditionalNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    std::string result = text_before
                         + condition->source_code(empty_indentation) 
                         + " ? " + yes->source_code(empty_indentation) 
                         + ", " + no->source_code(empty_indentation)
                         + text_after;
    return result;
}

ASTNodePtr ConditionalNode::get_condition() const{
    return condition;
}

ASTNodePtr ConditionalNode::get_yes() const{
    return yes;
}

ASTNodePtr ConditionalNode::get_no() const{
    return no;
}

void ConditionalNode::set_condition(const ASTNodePtr& _condition){
    condition = _condition;
}

void ConditionalNode::set_yes(const ASTNodePtr& _yes){
    yes = _yes;
}

void ConditionalNode::set_no(const ASTNodePtr& _no){
    no = _no;
}

std::string AssignmentNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    std::string result = text_before
                         + left_hand_operand->source_code(empty_indentation) 
                         + " " + op + " " 
                         + right_hand_operand->source_code(empty_indentation)
                         + text_after;
    return result;
}

std::string AssignmentNode::get_operator() const {
    return op;
}

void AssignmentNode::set_operator(const std::string& _operator) {
    op = _operator;
}

ASTNodePtr AssignmentNode::get_left_hand_operand() const {
    return left_hand_operand;    
}

ASTNodePtr AssignmentNode::get_right_hand_operand() const {
    return right_hand_operand;
}

void AssignmentNode::set_left_hand_operand(const ASTNodePtr& _operand) {
    left_hand_operand = _operand;
}

void AssignmentNode::set_right_hand_operand(const ASTNodePtr& _operand) {
    right_hand_operand = _operand;
}

std::string BreakNode::source_code(Indentation& _indentation) {
    visit(this);
    return text_before + _indentation + "break;" + text_after;
}

std::string ContinueNode::source_code(Indentation& _indentation) {
    visit(this);
    return text_before + _indentation + "continue;" + text_after;
}

std::string NewExpresionNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    return text_before + _indentation + "new " + type_name->source_code(empty_indentation) + text_after;
}

void NewExpresionNode::set_type_name(const ASTNodePtr& _type_name){
    type_name = _type_name;
}

ASTNodePtr NewExpresionNode::get_type_name() const{
    return type_name;
}

std::string EnumDefinitionNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    std::string result = text_before + _indentation + "enum " + name + "{";
    for (auto it = ast_nodes.begin(); it != ast_nodes.end(); ++it) {
        result = result + (*it)->source_code(empty_indentation) + ", ";
    }
    result = result.substr(0, result.length()-2) + "}" + text_after;
    return result;
}

void EnumDefinitionNode::add_member(const ASTNodePtr& _node) {
    append_sub_node(_node);
}

void EnumDefinitionNode::delete_member(const unsigned int& x) {
    delete_sub_node(x);
}

void EnumDefinitionNode::update_member(const unsigned int& x, const ASTNodePtr& _node) {
    update_sub_node(x, _node);
}

ASTNodePtr EnumDefinitionNode::get_member(const unsigned int& x) {
    return get_sub_node(x);
}

size_t EnumDefinitionNode::num_members() {
    return size();
}

void EnumDefinitionNode::set_name(const std::string& _name) {
    name = _name;
}

std::string EnumDefinitionNode::get_name() const {
    return name;
}

std::string EnumValueNode::source_code(Indentation& _indentation) {
    visit(this);
    return text_before + _indentation + name + text_after;
}

std::string EnumValueNode::get_name() const {
    return name;
}

void EnumValueNode::set_name(const std::string& _name) {
    name = _name;
}

std::string ThrowNode::source_code(Indentation& _indentation) {
    visit(this);
    return text_before + _indentation + "throw;" + text_after;
}

std::string PlaceHolderStatement::source_code(Indentation& _indentation) {
    visit(this);
    return text_before + _indentation + placeholder + ";" + text_after;
}

std::string PlaceHolderStatement::get_placeholder() const {
    return placeholder;
}

void PlaceHolderStatement::set_placeholder(const std::string& _placeholder) {
    placeholder = _placeholder;
}

std::string MappingNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    std::string result = text_before + _indentation 
                         + "mapping(" + key_type->source_code(empty_indentation) 
                         + " => " + value_type->source_code(empty_indentation) + ")"
                         + text_after;
    return result;
}

ASTNodePtr MappingNode::get_key_type() const{
    return key_type;
}

ASTNodePtr MappingNode::get_value_type() const{
    return value_type;
}

void MappingNode::set_key_type(const ASTNodePtr& _key_type){
    key_type = _key_type;
}

void MappingNode::set_value_type(const ASTNodePtr& _value_type){
    value_type = _value_type;
}

std::string ElementaryTypeNameNode::source_code(Indentation& _indentation) {
    visit(this);
    return text_before + _indentation + type_name + text_after;
}

void ElementaryTypeNameNode::set_type_name(const std::string& _type_name){
    type_name = _type_name;
}

std::string ElementaryTypeNameNode::get_type_name() const{
    return type_name;
}

std::string UserDefinedTypeNameNode::source_code(Indentation& _indentation) {
    visit(this);
    return text_before + _indentation + type_name + text_after;
}

void UserDefinedTypeNameNode::set_type_name(const std::string& _type_name){
    type_name = _type_name;
}

std::string UserDefinedTypeNameNode::get_type_name() const{
    return type_name;
}

std::string FunctionTypeNameNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    std::string result = text_before + _indentation + "function" + params->source_code(empty_indentation);
    if (returns->size()) result = result + " returns" + returns->source_code(empty_indentation);
    result = result + text_after;
    return result;
}

ASTNodePtr FunctionTypeNameNode::get_params() const{
    return params;
}

ASTNodePtr FunctionTypeNameNode::get_returns() const{
    return returns;
}

void FunctionTypeNameNode::set_params(const ASTNodePtr& _params){
    params = _params;
}

void FunctionTypeNameNode::set_returns(const ASTNodePtr& _returns){
    returns = _returns;
}

std::string ArrayTypeNameNode::source_code(Indentation& _indentation) {
    visit(this);
    Indentation empty_indentation(0);
    std::string result = text_before + _indentation + base_type->source_code(empty_indentation);
    if (size != nullptr) result = result + "[" + size->source_code(empty_indentation) + "]";
    else result = result + "[]";
    result = result + text_after;
    return result;

}

ASTNodePtr ArrayTypeNameNode::get_base_type() const{
    return base_type;
}

ASTNodePtr ArrayTypeNameNode::get_size() const{
    return size;
}

void ArrayTypeNameNode::set_base_type(const ASTNodePtr& _base_type){
    base_type = _base_type;
}

void ArrayTypeNameNode::set_size(const ASTNodePtr& _size){
    size = _size;
}

std::string InlineAssemblyNode::source_code(Indentation& _indentation) {
    visit(this);
    return text_before + _indentation + source + text_after;
}

void InlineAssemblyNode::set_source(std::string& _source) {
    source = _source;
    source = source.substr(0, source.find_last_of('}')+1);
}

std::string InlineAssemblyNode::get_source() {
    return source;
}



}
