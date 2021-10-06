#include <jimple-frontend/AST/jimple_ast.h>
#include <jimple-frontend/AST/jimple_type.h>

#pragma once
class jimple_expr : public jimple_ast {
public:

virtual exprt to_exprt() const {
    exprt val("at_identifier");
    return val;
};

};

class jimple_constant : public jimple_expr {
    public:
    //virtual std::string to_string() const override;
    virtual void from_json(const json& j) override;
    virtual std::string to_string() const override {
        return value;
    }
    virtual exprt to_exprt() const override;
    protected:
    std::string value;
};
/*
class jimple_new_expr : public jimple_expr {
    protected:
    std::vector<std::string> args;
    jimple_type t;
};
*/