/*
 * Parser.h
 *
 *  Created on: Jul 23, 2012
 *      Author: mikhail
 */

#ifndef PARSER_H_
#define PARSER_H_

#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

#include "clause.h"

class Parser {

public:
	Parser(const char* inFile);

	void printClauses();
	void printEquation();

private:
	string pretty(string line);
	string pretty(Clause c);

	vector<int> getVariables(string vars);

public:
	int num_of_vars;
	int num_of_clauses;
	vector<Clause> clauses;

private:
	ifstream file;
	bool has_p;
};

#endif /* PARSER_H_ */
