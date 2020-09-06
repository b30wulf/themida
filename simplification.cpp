#include "pch.h"

#include "simplification.hpp"
#include "ThemidaAnalyzer.hpp"
#include "ThemidaIR.hpp"
#include "Statement.hpp"
#include "BasicBlock.hpp"


IR::exprptr apply_xchg(IR::exprptr expression, const std::list<std::shared_ptr<IR::Instruction>>& xchg_list)
{
	switch (expression->get_type())
	{
		case IR::expr_register:
		case IR::expr_variable:
		{
			for (const auto& inst : xchg_list)
			{
				if (expression == inst->get_lhs())
					expression = inst->get_rhs();
				else if (expression == inst->get_rhs())
					expression = inst->get_lhs();
			}
			return expression;
		}
		case IR::expr_memory:
		{
			// maybe [base+index*scale+disp]?
			std::shared_ptr<IR::Memory> mem = std::dynamic_pointer_cast<IR::Memory>(expression);
			mem->set_expression(apply_xchg(mem->get_expression(), xchg_list));
			break;
		}
		case IR::expr_immediate:
		{
			break;
		}
		case IR::expr_unary_operation:
		{
			expression->set_operand(0, apply_xchg(expression->get_operand(0), xchg_list));
			break;
		}
		case IR::expr_binary_operation:
		{
			expression->set_operand(0, apply_xchg(expression->get_operand(0), xchg_list));
			expression->set_operand(1, apply_xchg(expression->get_operand(1), xchg_list));
			break;
		}
		default:
		{
			throw std::runtime_error("unknown expression");
			break;
		}
	}
	return expression;
}


// move xchg to bottom
void simplify_xchg(std::shared_ptr<IR::BB> bb)
{
	std::list<std::shared_ptr<IR::Instruction>> xchg_list;
	for (auto ho_it = bb->handler_objects.begin(); ho_it != bb->handler_objects.end(); ho_it++)
	{
		auto& handlers_instructions = ho_it->instructions;
		for (auto hi_it = handlers_instructions.begin(); hi_it != handlers_instructions.end();)
		{
			std::shared_ptr<IR::Instruction> inst = *hi_it;
			switch (inst->get_id())
			{
				case IR::instruction_id::assign:
				{
					inst->set_lhs(apply_xchg(inst->get_lhs(), xchg_list));
					inst->set_rhs(apply_xchg(inst->get_rhs(), xchg_list));
					break;
				}
				case IR::instruction_id::xchg:
				{
					// apply xchg first
					inst->set_lhs(apply_xchg(inst->get_lhs(), xchg_list));
					inst->set_rhs(apply_xchg(inst->get_rhs(), xchg_list));

					// then push_back
					xchg_list.push_back(inst);
					hi_it = handlers_instructions.erase(hi_it);
					continue;
				}
				case IR::instruction_id::push:
				case IR::instruction_id::pop:
				{
					inst->set_expression(apply_xchg(inst->get_expression(), xchg_list));
					break;
				}
				case IR::instruction_id::ret:
				case IR::instruction_id::jcc:	// unless..?
				case IR::instruction_id::jmp:	// unless..?
				case IR::instruction_id::xed:	// what to do
				case IR::instruction_id::lb:
				{
					break;
				}
				default:
				{
					throw std::runtime_error("unknown instruction");
				}
			}

			++hi_it;
		}
	}

	// append xchg instructions (reverse), should be able to remove all tho
	IR::handler_instructions last;
	for (auto it = xchg_list.rbegin(); it != xchg_list.rend(); ++it)
	{
		last.instructions.push_back(*it);
	}
	bb->handler_objects.push_back(last);
}


void simplify_instructions(std::list<std::shared_ptr<IR::Instruction>>& instructions, bool basic_block)
{
	// fix later
	auto is_temp_variable = [basic_block](const IR::exprptr& expression) -> bool
	{
		if (expression && expression->get_type() == IR::expr_variable)
		{
			std::shared_ptr<IR::Variable> _var = std::dynamic_pointer_cast<IR::Variable>(expression);
			if (/*basic_block || */_var->get_name().find("t") == 0)
				return true;
		}
		return false;
	};

	// reference count
	std::function<void(const IR::exprptr&, std::map<IR::exprptr, int>&)>
		inc_reference_count = [&inc_reference_count](const IR::exprptr& expression,
			std::map<IR::exprptr, int>& refcount)
	{
		switch (expression->get_type())
		{
			case IR::expr_register:
			case IR::expr_variable:
			{
				refcount[expression] += 1;
				break;
			}
			case IR::expr_memory:
			{
				// maybe [base+index*scale+disp]?
				std::shared_ptr<IR::Memory> mem = std::dynamic_pointer_cast<IR::Memory>(expression);
				inc_reference_count(mem->get_expression(), refcount);
				break;
			}
			case IR::expr_immediate:
			{
				break;
			}
			case IR::expr_unary_operation:
			{
				inc_reference_count(expression->get_operand(0), refcount);
				break;
			}
			case IR::expr_binary_operation:
			{
				inc_reference_count(expression->get_operand(0), refcount);
				inc_reference_count(expression->get_operand(1), refcount);
				break;
			}
			default:
			{
				throw std::runtime_error("unknown expression");
				break;
			}
		}
	};

	// dec_reference_count
	std::function<void(const IR::exprptr&, std::map<IR::exprptr, int>&)>
		dec_reference_count = [&dec_reference_count](const IR::exprptr& expression, std::map<IR::exprptr, int>& refcount)
	{
		switch (expression->get_type())
		{
			case IR::expr_register:
			case IR::expr_variable:
			{
				auto it = refcount.find(expression);
				if (it != refcount.end())
				{
					--it->second;
					if (it->second < 0)
						it->second = 0;
				}
				break;
			}
			case IR::expr_memory:
			{
				// maybe [base+index*scale+disp]?
				std::shared_ptr<IR::Memory> mem = std::dynamic_pointer_cast<IR::Memory>(expression);
				dec_reference_count(mem->get_expression(), refcount);
				break;
			}
			case IR::expr_immediate:
			{
				break;
			}
			case IR::expr_unary_operation:
			{
				dec_reference_count(expression->get_operand(0), refcount);
				break;
			}
			case IR::expr_binary_operation:
			{
				dec_reference_count(expression->get_operand(0), refcount);
				dec_reference_count(expression->get_operand(1), refcount);
				break;
			}
			default:
			{
				throw std::runtime_error("unknown expression");
				break;
			}
		}
	};

	//
	// remove unused temp variable or "assign X=X"
	//
	std::map<IR::exprptr, int> refcount;
	for (auto it = instructions.rbegin(); it != instructions.rend();)
	{
		std::shared_ptr<IR::Instruction> instruction = *it;
		switch (instruction->get_id())
		{
			case IR::instruction_id::assign:
			{
				const auto lhs = instruction->get_lhs();
				const auto rhs = instruction->get_rhs();
				if (lhs == rhs || (is_temp_variable(lhs) && refcount[lhs] == 0))
				{
					// remove unused temp variable
					//dec_reference_count(rhs, refcount); // i mean... you didn't inc so no need to dec count lol
					instructions.erase(--(it.base()));
					continue;
				}
				else
				{
					refcount[lhs] = 0;
					if (lhs->get_type() == IR::expr_memory)
					{
						std::shared_ptr<IR::Memory> mem = std::dynamic_pointer_cast<IR::Memory>(lhs);
						inc_reference_count(mem->get_expression(), refcount);
					}
					inc_reference_count(rhs, refcount);
				}
				break;
			}
			case IR::instruction_id::xchg:
			{
				inc_reference_count(instruction->get_lhs(), refcount);
				inc_reference_count(instruction->get_rhs(), refcount);
				break;
			}
			case IR::instruction_id::push:
			case IR::instruction_id::pop:
			{
				inc_reference_count(instruction->get_expression(), refcount);
				break;
			}
			case IR::instruction_id::ret:
			case IR::instruction_id::jcc:	// unless..?
			case IR::instruction_id::jmp:	// unless..?
			{
				break;
			}
			case IR::instruction_id::xed:	// what to do
			{
				break;
			}
			case IR::instruction_id::lb:
			{
				break;
			}
			default:
			{
				throw std::runtime_error("unknown instruction");
				break;
			}
		}

		++it;
	}

	if (basic_block)
		return;

	//
	// const propagation
	//
	std::map<IR::exprptr, IR::exprptr> assigned;
	std::set<IR::exprptr> written;
	std::function<bool(const IR::exprptr&)> is_written = [&is_written, &written](const IR::exprptr& expression) -> bool
	{
		switch (expression->get_type())
		{
			case IR::expr_register:
			case IR::expr_variable:
			{
				return written.count(expression) > 0;
			}
			case IR::expr_memory:
			{
				std::shared_ptr<IR::Memory> mem = std::dynamic_pointer_cast<IR::Memory>(expression);
				return is_written(mem->get_expression());
			}
			case IR::expr_immediate:
			{
				return false;
			}
			case IR::expr_unary_operation:
			{
				return is_written(expression->get_operand(0));
			}
			case IR::expr_binary_operation:
			{
				return is_written(expression->get_operand(0)) || is_written(expression->get_operand(1));
			}
			default:
			{
				throw std::runtime_error("unknown expression");
			}
		}
	};
	std::function<IR::exprptr(const IR::exprptr&)> propagation = [&propagation, &is_written, &assigned, &written](
		const IR::exprptr& expression) -> IR::exprptr
	{
		switch (expression->get_type())
		{
			case IR::expr_register:
			case IR::expr_variable:
			{
				auto assigned_it = assigned.find(expression);
				if (assigned_it != assigned.end() && !is_written(assigned_it->second))
				{
					// set and deref
					return assigned_it->second;
				}
				break;
			}
			case IR::expr_memory:
			{
				// maybe [base+index*scale+disp]?
				std::shared_ptr<IR::Memory> mem = std::dynamic_pointer_cast<IR::Memory>(expression);
				mem->set_expression(propagation(mem->get_expression()));
				break;
			}
			case IR::expr_immediate:
			{
				break;
			}
			case IR::expr_unary_operation:
			{
				expression->set_operand(0, propagation(expression->get_operand(0)));
				break;
			}
			case IR::expr_binary_operation:
			{
				expression->set_operand(0, propagation(expression->get_operand(0)));
				expression->set_operand(1, propagation(expression->get_operand(1)));
				break;
			}
			default:
			{
				throw std::runtime_error("unknown expression");
				break;
			}
		}
		return expression;
	};
	for (auto it = instructions.begin(); it != instructions.end();)
	{
		std::shared_ptr<IR::Instruction> instruction = *it;
		switch (instruction->get_id())
		{
			case IR::instruction_id::assign:
			{
				auto lhs = instruction->get_lhs();

				instruction->set_rhs(simplify_expression(propagation(instruction->get_rhs())));
				written.insert(lhs);

				assigned[lhs] = instruction->get_rhs();
				break;
			}
			case IR::instruction_id::xchg:
			{
				written.insert(instruction->get_lhs());
				written.insert(instruction->get_rhs());
				break;
			}
			case IR::instruction_id::push:
			case IR::instruction_id::pop:
			{
				break;
			}
			case IR::instruction_id::ret:
			case IR::instruction_id::jcc:	// unless..?
			case IR::instruction_id::jmp:	// unless..?
			{
				break;
			}
			case IR::instruction_id::xed:	// what to do
			{
				break;
			}
			case IR::instruction_id::lb:
			{
				break;
			}
			default:
			{
				throw std::runtime_error("unknown instruction");
				break;
			}
		}

		++it;
	}


	// x=y <- xchg
	// y=z
	// z=x
	std::vector<std::list<std::shared_ptr<IR::Instruction>>::iterator> assigned_list;
	//assigned_list.reserve(3);
	for (auto it = instructions.begin(); it != instructions.end();)
	{
		std::shared_ptr<IR::Instruction> instruction = *it;
		switch (instruction->get_id())
		{
			case IR::instruction_id::assign:
			{
				auto lhs = instruction->get_lhs();
				auto rhs = instruction->get_rhs();
				assigned_list.push_back(it);
				if (assigned_list.size() == 3)
				{
					auto _it1 = *assigned_list[0];
					auto _it2 = *assigned_list[1];
					auto _it3 = *assigned_list[2];
					if (_it1->get_lhs() == _it3->get_rhs()
						&& _it1->get_rhs() == _it2->get_lhs()
						&& _it2->get_rhs() == _it3->get_lhs())
					{
						// replace with xchg
						instructions.erase(assigned_list[0]);	// maybe better erase
						instructions.erase(assigned_list[1]);	// maybe better erase
						it = instructions.erase(assigned_list[2]);	// maybe better erase
						it = instructions.insert(it, std::make_shared<IR::Xchg>(_it2->get_lhs(), _it2->get_rhs()));
						assigned_list.clear();
						continue;
					}
					else
					{
						// remove first assign, vector isn't good tbh
						assigned_list.erase(assigned_list.begin());
					}
				}
				break;
			}
			default:
			{
				assigned_list.clear();
				break;
			}
		}

		++it;
	}
}