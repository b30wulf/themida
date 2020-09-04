#include "pch.h"

#include "Statement.hpp"
#include "ThemidaIR.hpp"

namespace IR
{
	// Instruction
	Instruction::Instruction(instruction_id id) : m_instruction_id(id)
	{
	}
	Instruction::~Instruction()
	{
	}
	instruction_id Instruction::get_id() const
	{
		return this->m_instruction_id;
	}

	// Assign
	Assign::Assign(const exprptr& lhs, const exprptr& rhs) : Instruction(instruction_id::assign)
	{
		this->m_expressions.push_back(lhs);
		this->m_expressions.push_back(rhs);
	}
	void Assign::to_string(std::ostream& stream) const
	{
		stream << "mov " << this->m_expressions[0] << "," << this->m_expressions[1];
	}


	// Push
	Push::Push(const exprptr&expr) : Instruction(instruction_id::push)
	{
		this->m_expressions.push_back(expr);
	}
	void Push::to_string(std::ostream& stream) const
	{
		stream << "push (" << this->m_expressions[0] << ")";
	}


	// Pop
	Pop::Pop(const exprptr&expr) : Instruction(instruction_id::pop)
	{
		this->m_expressions.push_back(expr);
	}
	void Pop::to_string(std::ostream& stream) const
	{
		stream << "pop (" << this->m_expressions[0] << ")";
	}


	// Xchg
	Xchg::Xchg(const exprptr& lhs, const exprptr& rhs) : Instruction(instruction_id::xchg)
	{
		this->m_expressions.push_back(lhs);
		this->m_expressions.push_back(rhs);
	}
	void Xchg::to_string(std::ostream& stream) const
	{
		stream << "xchg " << this->m_expressions[0] << "," << this->m_expressions[1];
	}


	std::ostream& operator<<(std::ostream& stream, const Instruction& statement)
	{
		statement.to_string(stream);
		return stream;
	}
	std::ostream& operator<<(std::ostream& stream, const Instruction* statement)
	{
		statement->to_string(stream);
		return stream;
	}
}