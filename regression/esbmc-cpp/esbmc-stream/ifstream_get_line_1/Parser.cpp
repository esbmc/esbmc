/*
 * Parser.cpp
 *
 *  Created on: Jul 23, 2012
 *      Author: mikhail
 */

#include "Parser.h"

#include <cstdlib>
#include <cctype>
#include <sstream>
#include <string>

Parser::Parser(const char* inFile)
  : num_of_vars(-1),
		num_of_clauses(-1),
    file(inFile),
    has_p(false)
{
	if(file)
	{
		string line;
		int clauses_aux = 0;

		cout << boolalpha;
		while (file.good())
		{
			getline(file, line);

			if((*line.begin()) == 'c') // comentario
			{
				continue;
			}
			else if((*line.begin()) == 'p') // começa o programa
			{
				if(has_p)
					throw 2;

				has_p = true;

				// p
				string aux = line;
				string elem = aux.substr(0, aux.find_first_of(" "));

				// cnf
				aux = aux.substr(aux.find_first_of(" ")+1);
				elem = aux.substr(0, aux.find_first_of(" "));

				if(elem != "cnf")
					throw 1;

				// num_of_vars
				aux = aux.substr(aux.find_first_of(" ")+1);
				elem = aux.substr(0, aux.find_first_of(" "));

				num_of_vars = atoi(elem.c_str());

				// num_of_clauses
				aux = aux.substr(aux.find_first_of(" ")+1);
				elem = aux.substr(0, aux.find_first_of(" "));

				num_of_clauses = atoi(elem.c_str());

				if(num_of_vars < 1 or num_of_clauses < 1)
					throw 1;
			}
			else
			{
				string aux = line;
				string elem = aux.substr(0, aux.find_first_of(" "));

				if(elem.size() > 1 and elem.at(0)=='-') // negativo?
					elem = aux.substr(1,elem.length()-1);

				if(elem != "")
				{
					if(!has_p)
						throw 3;

					if(clauses_aux >= num_of_clauses)
					{
						cout << "Ignorando clausula " << clauses_aux+1 << ", somente " << num_of_clauses << " foram definidas." << endl;
						cout << line << endl << endl;
						++clauses_aux;
						continue;
					}

					aux = aux.substr(0, aux.find_last_of("0"));

					Clause c;
					c.variables = getVariables(aux);

					clauses.push_back(c);
					++clauses_aux;
				}
				else
				{
					// linha em branco ou fim do arquivo?
					continue;
				}
			}
		}

		if(clauses_aux < num_of_clauses)
			throw 4;

		file.close();
	}
	else
		throw 0;
}

vector<int> Parser::getVariables(string vars)
{
	vector<int> out;

	string aux = vars;
	string elem = aux.substr(0, aux.find_first_of(" "));

	do
	{
		if(elem != "")
			out.push_back(atoi(elem.c_str()));

		aux = aux.substr(aux.find_first_of(" ")+1);
		elem = aux.substr(0, aux.find_first_of(" "));

		if(aux == "")
			break;
	} while(1);

	return out;
}

void Parser::printClauses()
{
	for(unsigned int i=0; i < clauses.size(); ++i)
		cout << "Clausula " << i+1 << ": " << pretty(clauses.at(i)) << endl;
}

void Parser::printEquation()
{
	cout << "Equation: " << endl;
	unsigned int i=0;
	for(i=0; i < clauses.size()-1; ++i)
		cout << pretty(clauses.at(i)) << " and ";
	cout << pretty(clauses.at(i)) << endl;
}

string Parser::pretty(Clause c)
{
	stringstream out;
	out << "(";

	unsigned int i=0;
	do
	{
		int elem = c.variables.at(i);

		if(elem < 0)
		{
			out << " ( NOT";
			out << " x(" << (-1)*elem << ") ";
			out << ") ";
		}
		else
		{
			out << " x(" << elem << ") ";
		}

		++i;

		if(i<c.variables.size())
			out << "OR";

	} while (i<c.variables.size());


	out << ")";

	return out.str();
}

string Parser::pretty(string line)
{
	string aux = line;
	string elem = aux.substr(0, aux.find_first_of(" "));

	string out = "(";
	do
	{
		bool has_not = false;
		if(elem.size() > 1 and elem.at(0)=='-')
		{
			out += " ( NOT";
			elem = aux.substr(1,elem.length()-1);
			has_not = true;
		}

		if(elem != "")
			out += " x(" + elem + ") ";

		if(has_not)
			out += ") ";

		aux = aux.substr(aux.find_first_of(" ")+1);
		elem = aux.substr(0, aux.find_first_of(" "));

		if(aux == "")
			break;

		if(elem != "")
			out += "OR";
	} while(1);

	out += ")";

	return out;
}
