#include "pch.h"

#include "tritonhelper.hpp"

triton::engines::symbolic::SharedSymbolicVariable get_symbolic_var(const triton::ast::SharedAbstractNode &node)
{
	return node->getType() == triton::ast::VARIABLE_NODE ?
		std::dynamic_pointer_cast<triton::ast::VariableNode>(node)->getSymbolicVariable() : nullptr;
}
std::set<triton::ast::SharedAbstractNode> collect_symvars(const triton::ast::SharedAbstractNode &parent)
{
	std::set<triton::ast::SharedAbstractNode> result;
	if (!parent)
		return result;

	if (parent->getChildren().empty() && parent->isSymbolized())
	{
		// this must be variable node right?
		assert(parent->getType() == triton::ast::VARIABLE_NODE);
		result.insert(parent);
	}

	for (const triton::ast::SharedAbstractNode &child : parent->getChildren())
	{
		if (!child->getChildren().empty())
		{
			// go deep if symbolized
			if (child->isSymbolized())
			{
				auto _new = collect_symvars(child);
				result.insert(_new.begin(), _new.end());
			}
		}
		else if (child->isSymbolized())
		{
			// this must be variable node right?
			assert(child->getType() == triton::ast::VARIABLE_NODE);
			result.insert(child);
		}
	}
	return result;
}


bool is_unary_operation(const triton::arch::Instruction &triton_instruction)
{
	switch (triton_instruction.getType())
	{
		case triton::arch::x86::ID_INS_INC:
		case triton::arch::x86::ID_INS_DEC:
		case triton::arch::x86::ID_INS_NEG:
		case triton::arch::x86::ID_INS_NOT:
			return true;

		default:
			return false;
	}
}
bool is_binary_operation(const triton::arch::Instruction &triton_instruction)
{
	switch (triton_instruction.getType())
	{
		case triton::arch::x86::ID_INS_ADD:
		case triton::arch::x86::ID_INS_SUB:
		case triton::arch::x86::ID_INS_SHL:
		case triton::arch::x86::ID_INS_SHR:
		case triton::arch::x86::ID_INS_RCR:
		case triton::arch::x86::ID_INS_RCL:
		case triton::arch::x86::ID_INS_ROL:
		case triton::arch::x86::ID_INS_ROR:
		case triton::arch::x86::ID_INS_AND:
		case triton::arch::x86::ID_INS_OR:
		case triton::arch::x86::ID_INS_XOR:
		case triton::arch::x86::ID_INS_CMP:
		case triton::arch::x86::ID_INS_TEST:
			return true;

		case triton::arch::x86::ID_INS_IMUL:
		{
			// imul can have 3 operands but eh
			return triton_instruction.operands.size() == 2;
		}

		default:
			return false;
	}
}
bool is_mov_operation(const triton::arch::Instruction& triton_instruction)
{
	switch (triton_instruction.getType())
	{
		case triton::arch::x86::ID_INS_MOV:
		case triton::arch::x86::ID_INS_MOVSX:
		case triton::arch::x86::ID_INS_MOVZX:
			return true;

		default:
			return false;
	}
}