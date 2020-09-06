#include "pch.h"

#include "ThemidaAnalyzer.hpp"
#include "ThemidaIR.hpp"
#include "Statement.hpp"
#include "x86_instruction.hpp"
#include "AbstractStream.hpp"
#include "CFG.hpp"
#include "tritonhelper.hpp"
#include "BasicBlock.hpp"
#include "simplification.hpp"


constexpr bool print_debug = false;
constexpr bool strict_check = false;
static AbstractStream *g_stream = nullptr;


// just make it BV_NODE if node gets too complicated
static triton::ast::SharedAbstractNode not_too_complicated(triton::API& api, const triton::ast::SharedAbstractNode& node)
{
	if (node->getLevel() >= 30 && node->isSymbolized())
	{
		return api.getAstContext()->bv(node->evaluate(), node->getBitvectorSize());
	}
	return node;
}


// read from stream (this is faster than concretize entire memory perhaps)
static void try_reading_memory(triton::API& ctx, const triton::arch::MemoryAccess& mem)
{
	const triton::uint64 address = mem.getAddress();
	if (!ctx.isConcreteMemoryValueDefined(mem.getAddress(), mem.getSize()))
	{
		const auto previous_pos = g_stream->pos();
		g_stream->seek(mem.getAddress());

		std::vector<triton::uint8> values;
		const triton::uint32 mem_size = std::min<triton::uint32>(mem.getSize(), 1024);
		values.resize(mem_size);
		if (g_stream->read(&values[0], mem_size) == mem_size)
		{
			ctx.setConcreteMemoryAreaValue(address, values);
		}
		else
		{
			/*std::stringstream ss;
			ss << "Failed to read memory at 0x" << std::hex << address << "\n"
				<< "\tFile: " << __FILE__ << ", L: " << __LINE__;
			throw std::runtime_error(ss.str());*/
		}

		g_stream->seek(previous_pos);
	}
}


// functions to check memory address
static bool is_lea_tainted(triton::API& ctx, const triton::arch::MemoryAccess& mem)
{
	constexpr bool check_segment_register = false;
	return ctx.isRegisterTainted(mem.getConstBaseRegister())
		|| ctx.isRegisterTainted(mem.getConstIndexRegister())
		|| (check_segment_register && ctx.isRegisterTainted(mem.getConstSegmentRegister()));
}
static bool is_bytecode_address(const triton::ast::SharedAbstractNode &lea_ast, std::shared_ptr<ThemidaHandlerContext> context)
{
	if (strict_check)
	{
		// return true if lea_ast is constructed by bytecode
		const std::set<triton::ast::SharedAbstractNode> symvars = collect_symvars(lea_ast);
		if (symvars.empty())
			return false;

		for (auto it = symvars.begin(); it != symvars.end(); ++it)
		{
			const triton::ast::SharedAbstractNode &node = *it;
			const triton::engines::symbolic::SharedSymbolicVariable &symvar = std::dynamic_pointer_cast<triton::ast::VariableNode>(node)->getSymbolicVariable();
			if (symvar->getId() != context->symvar_bytecode->getId())
				return false;
		}
	}

	// I assume max bytecode is 0x50
	const triton::uint64 runtime_address = lea_ast->evaluate().convert_to<triton::uint64>();
	return context->bytecode <= runtime_address && runtime_address < (context->bytecode + 0x50);
}
static bool is_stack_address(const triton::ast::SharedAbstractNode &lea_ast, std::shared_ptr<ThemidaHandlerContext> context)
{
	// return true if lea_ast is constructed by stack
	const std::set<triton::ast::SharedAbstractNode> symvars = collect_symvars(lea_ast);
	if (symvars.empty())
		return false;

	for (auto it = symvars.begin(); it != symvars.end(); ++it)
	{
		const triton::ast::SharedAbstractNode &node = *it;
		const triton::engines::symbolic::SharedSymbolicVariable &symvar = std::dynamic_pointer_cast<triton::ast::VariableNode>(node)->getSymbolicVariable();
		if (symvar != context->symvar_stack)
			return false;
	}
	return true;
}
static bool is_context_address(const triton::ast::SharedAbstractNode &lea_ast, std::shared_ptr<ThemidaHandlerContext> context)
{
	// just check runtime address for now
	const triton::uint64 runtime_address = lea_ast->evaluate().convert_to<triton::uint64>();
	return context->context <= runtime_address && runtime_address < (context->context + THEMIDA_CONTEXT_SIZE);
}
static bool is_vm_reg(const triton::ast::SharedAbstractNode &lea_ast, std::shared_ptr<ThemidaHandlerContext> context)
{
	const triton::engines::symbolic::SharedSymbolicVariable symvar = get_symbolic_var(lea_ast);
	return symvar && context->vmregs.find(symvar->getId()) != context->vmregs.end();
}


// analysis
bool ThemidaAnalyzer::symbolize_read_memory(const triton::arch::MemoryAccess& mem, std::shared_ptr<ThemidaHandlerContext> context)
{
	const triton::uint64 address = mem.getAddress();
	triton::ast::SharedAbstractNode lea_ast = mem.getLeaAst();
	if (!lea_ast || !lea_ast->isSymbolized())
	{
		try_reading_memory(*triton_api, mem);
		return false;
	}

	lea_ast = triton_api->processSimplification(lea_ast, true);
	if (is_bytecode_address(lea_ast, context))
	{
		try_reading_memory(*triton_api, mem);

		// taint bytecode
		triton_api->taintMemory(mem);
	}

	// lea_ast = context + const
	else if (is_context_address(lea_ast, context))
	{
		// the instruction loads data from themida context [BP+OFFSET]
		//
		const triton::uint64 offset = lea_ast->evaluate().convert_to<triton::uint64>() - context->context;
		if (is_lea_tainted(*triton_api, mem))
		{
			// declare temp
			auto temp = IR::Variable::create_variable(mem.getSize());

			// symbolize memory as VM_REGISTER
			triton::engines::symbolic::SharedSymbolicVariable symvar_vmreg = triton_api->symbolizeMemory(mem, temp->get_name());
			context->vmregs.insert(std::make_pair(symvar_vmreg->getId(), symvar_vmreg));
			if (print_debug)
			std::cout << "Load [EBP+0x" << std::hex << offset << "](VM_REGISTER)\n";

			// temp = VM_REG
			auto _source = this->get_vm_register(offset, mem.getSize());
			context->instructions.push_back(std::make_shared<IR::Assign>(temp, _source));
			context->m_expression_map[symvar_vmreg->getId()] = temp;

			return true;
		}
		else if (address == context->bytecode)
		{
			if (print_debug)
			std::cout << "Load bytecode address\n";
		}
		else
		{
			// maybe create VM_TEMP_VARIABLE here? so we can trace stuff between handlers (specifically for dolphinVM)
			if (print_debug)
			std::cout << "Load [EBP+0x" << std::hex << offset << "](STATIC)\n";
		}

		const triton::uint64 vm_jcc_offset
			//= 0x111; // fish64 white
			= 0xbb; // fish32 white
			//= 0x1e; // tiger32 w
			//= 0xcb; // tiger64 white
		if (offset == vm_jcc_offset)
		{
			triton_api->setConcreteMemoryValue(mem, context->execute_jcc);
			context->jcc_detected = true;
		}
	}

	// can ignore stack address
	else if (false && is_stack_address(lea_ast, context))
	{
		//const triton::uint64 offset = address - context->stack;
		//std::cout << "Access [ESP+0x" << std::hex << offset << "]" << "\n";
	}

	// deref(vm)
	else if (is_vm_reg(lea_ast, context))
	{
		// temp = deref(vm_reg)
		// symbolize memory as temp

		triton::arch::Register segment_register = mem.getConstSegmentRegister();
		if (segment_register.getId() == triton::arch::ID_REG_INVALID)
		{
			// data segment if invalid?
			segment_register = triton_api->registers.x86_ds;
		}

		// load IR expression by triton symvar
		triton::engines::symbolic::SharedSymbolicVariable symvar_source = get_symbolic_var(lea_ast);
		auto it = context->m_expression_map.find(symvar_source->getId());
		if (it == context->m_expression_map.end())
			throw std::runtime_error("very unexpected");
		std::shared_ptr<IR::Expression> expr = it->second;

		// declare temp
		auto temp = IR::Variable::create_variable(mem.getSize());

		// symbolize memory
		const triton::engines::symbolic::SharedSymbolicVariable symvar = triton_api->symbolizeMemory(mem, temp->get_name());
		if (print_debug)
		std::cout << "Deref(" << lea_ast << "," << segment_register.getName() << ")\n";

		// temp = deref(expr)
		std::shared_ptr<IR::Expression> _memory = std::make_shared<IR::Memory>(expr, segment_register, mem.getSize());
		context->instructions.push_back(std::make_shared<IR::Assign>(temp, _memory));
		context->m_expression_map[symvar->getId()] = temp;

		return true;
	}
	else
	{
		try_reading_memory(*triton_api, mem);
		if (print_debug)
		std::cout << "unknown read addr: " << std::hex << address << " " << lea_ast << "\n";
	}

	return false;
}
void ThemidaAnalyzer::storeAccess(triton::arch::Instruction &triton_instruction, std::shared_ptr<ThemidaHandlerContext> context)
{
	const std::set<std::pair<triton::arch::MemoryAccess, triton::ast::SharedAbstractNode>>& storeAccess = triton_instruction.getStoreAccess();
	for (const std::pair<triton::arch::MemoryAccess, triton::ast::SharedAbstractNode>& pair : storeAccess)
	{
		const triton::arch::MemoryAccess &mem = pair.first;
		//const triton::ast::SharedAbstractNode &mem_ast = pair.second;
		const triton::ast::SharedAbstractNode mem_ast = triton_api->getMemoryAst(mem);
		const triton::uint64 address = mem.getAddress();
		if (address == context->lock_addr)
		{
			if (print_debug)
			std::cout << "vm exit detected" << "\n";
			context->exit_detected = true;
		}

		triton::ast::SharedAbstractNode lea_ast = mem.getLeaAst();
		if (!lea_ast)
		{
			// most likely can be ignored
			continue;
		}

		lea_ast = triton_api->processSimplification(lea_ast, true);
		if (!lea_ast->isSymbolized())
		{
			// most likely can be ignored
			continue;
		}

		if (address == context->bytecode_addr)
		{
			// move themida-instruction-pointer
			const triton::uint64 bytecode = triton_api->getConcreteMemoryValue(mem).convert_to<triton::uint64>();
			if (print_debug)
			std::cout << "bytecode=0x" << std::hex << bytecode << "\n";
			if (triton_api->isMemoryTainted(mem))
			{
				// if bytecode comes from tainted(bytecode) it should be jmp(jcc) inside
				context->jmp_inside_detected = true;
			}
		}
		else if (is_context_address(lea_ast, context))
		{
			// the instruction writes [BP+OFFSET]
			// 
			const triton::uint64 offset = lea_ast->evaluate().convert_to<triton::uint64>() - context->context;
			if (is_lea_tainted(*triton_api, mem))
			{
				// VM_REG_X = Y
				if (print_debug)
				std::cout << "Store VM_REG [EBP+0x" << std::hex << offset << "]\n";

				// create IR (VM_REG = mem_ast)
				auto source_node = triton_api->processSimplification(mem_ast, true);
				if (source_node->getType() == triton::ast::BV_NODE)
				{
					// VM_REG_X = immediate
					std::shared_ptr<IR::Expression> v1 = this->get_vm_register(offset, mem.getSize()); // dont inc count here
					std::shared_ptr<IR::Instruction> _assign = std::make_shared<IR::Assign>(v1, std::make_shared<IR::Immediate>(
						source_node->evaluate().convert_to<triton::uint64>()));
					context->instructions.push_back(_assign);
					continue;
				}

				triton::engines::symbolic::SharedSymbolicVariable symvar = get_symbolic_var(source_node);
				if (symvar)
				{
					std::shared_ptr<IR::Expression> v1 = this->get_vm_register(offset, mem.getSize());	// dont inc count here
					auto it = context->m_expression_map.find(symvar->getId());
					if (it != context->m_expression_map.end())
					{
						// VM_REG_X = expression
						std::shared_ptr<IR::Expression> expr = it->second;
						context->instructions.push_back(std::make_shared<IR::Assign>(v1, expr));
					}
					else if (symvar->getAlias().find("topofstack") != std::string::npos) // nice try ^_^
					{
						// VM_REG_X = topofstack
						std::shared_ptr<IR::Expression> expr = std::make_shared<IR::Variable>("TopOfStack", mem.getSize());
						context->instructions.push_back(std::make_shared<IR::Assign>(v1, expr));
					}
					else
					{
						printf("%s\n", symvar->getAlias().c_str());
						throw std::runtime_error("what do you mean 2");
					}
				}
				else
				{
					std::cout << mem << "\n";
					std::cout << "source_node: " << source_node << "\n";
					throw std::runtime_error("storeAccess source_node error?");
				}
			}
			else
			{
				if (print_debug)
				std::cout << "Store [EBP+0x" << std::hex << offset << "](STATIC)\n";
				const auto val = triton_api->getConcreteMemoryValue(mem).convert_to<triton::uint64>();
				context->static_written[offset] = { mem.getSize(), val };
			}
		}
		else if (false && is_stack_address(lea_ast, context))
		{
			// stores to stack
		}
		else
		{
			// create IR (VM_REG = mem_ast)
			// get right expression
			std::shared_ptr<IR::Expression> expr;
			auto simplified_source_node = triton_api->processSimplification(mem_ast, true);
			if (!simplified_source_node->isSymbolized())
			{
				// expression is immediate
				expr = std::make_shared<IR::Immediate>(simplified_source_node->evaluate().convert_to<triton::uint64>());
			}
			else
			{
				// ignore ebp as it's useless
				triton::engines::symbolic::SharedSymbolicVariable symvar1 = get_symbolic_var(simplified_source_node);
				if (symvar1 && symvar1->getId() != context->symvar_context->getId())
				{
					auto _it = context->m_expression_map.find(symvar1->getId());
					if (_it == context->m_expression_map.end())
					{
						std::cout << symvar1 << "\n";
						throw std::runtime_error("what do you mean...");
					}
					expr = _it->second;
				}
			}

			triton::engines::symbolic::SharedSymbolicVariable symvar0 = get_symbolic_var(lea_ast);
			if (symvar0 && expr)
			{
				auto it0 = context->m_expression_map.find(symvar0->getId());
				if (it0 != context->m_expression_map.end())
				{
					triton::arch::Register segment_register = mem.getConstSegmentRegister();
					if (segment_register.getId() == triton::arch::ID_REG_INVALID)
					{
						// DS?
						segment_register = triton_api->registers.x86_ds;
					}
					std::shared_ptr<IR::Expression> v1 = std::make_shared<IR::Memory>(it0->second, segment_register, mem.getSize());
					context->instructions.push_back(std::make_shared<IR::Assign>(v1, expr));
					context->runtime_memory[v1] = address;
				}
				else
				{
					throw std::runtime_error("what do you mean 2");
				}
			}
			else
			{
				if (print_debug)
				std::cout << "unknown store addr: " << std::hex << address << ", lea_ast: " << lea_ast << ", simplified_source_node: " << simplified_source_node << "\n";
			}
		}
	}
}

// save expressions if any operands are IR::Expression
std::vector<std::shared_ptr<IR::Expression>> ThemidaAnalyzer::save_expressions(
	triton::arch::Instruction &triton_instruction, std::shared_ptr<ThemidaHandlerContext> context)
{
	std::vector<std::shared_ptr<IR::Expression>> expressions;
	if (!is_unary_operation(triton_instruction)
		&& !is_binary_operation(triton_instruction)
		&& !is_mov_operation(triton_instruction))
	{
		return expressions;
	}

	bool _save = false;
	const std::vector<triton::arch::OperandWrapper>& operands = triton_instruction.operands;
	for (size_t i = 0; i < operands.size(); i++)
	{
		const triton::arch::OperandWrapper& operand = operands[i];
		if (is_mov_operation(triton_instruction) && i == 0)
		{
			// no need to save lhs with mov|movsx|movzx
			continue;
		}

		if (operand.getType() == triton::arch::OP_IMM)
		{
			const triton::arch::Immediate& _imm = operand.getConstImmediate();
			expressions.push_back(std::make_shared<IR::Immediate>(_imm.getValue(), _imm.getSize()));
		}
		else if (operand.getType() == triton::arch::OP_MEM)
		{
			const triton::arch::MemoryAccess& _mem = operand.getConstMemory();
			triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(triton_api->processSimplification(triton_api->getMemoryAst(_mem), true));
			if (_symvar)
			{
				if (_symvar->getId() == context->symvar_bytecode->getId())
				{
					// ignore bytecode
					continue;
				}

				// load symbolic
				auto _it = context->m_expression_map.find(_symvar->getId());
				if (_it != context->m_expression_map.end())
				{
					std::shared_ptr<IR::Expression> _expr = _it->second;
					if (_mem.getSize() == _it->second->get_size())
					{
						expressions.push_back(_expr);
					}
					else
					{
						expressions.push_back(std::make_shared<IR::Extract>(_it->second, _mem.getBitSize(), 0));
					}
					_save = true;
					continue;
				}
			}

			// otherwise immediate
			expressions.push_back(std::make_shared<IR::Immediate>(
				triton_api->getConcreteMemoryValue(_mem).convert_to<triton::uint64>()));
		}
		else if (operand.getType() == triton::arch::OP_REG)
		{
			bool _make_imm = true;
			const triton::arch::Register& op_reg = operand.getConstRegister();
			std::vector<triton::arch::Register> overlapped_regs = this->get_overlapped_regs(op_reg);
			for (auto overlapped_reg : overlapped_regs)
			{
				auto it = context->known_regs.find(overlapped_reg);
				if (it != context->known_regs.end())
				{
					// reg is known
					triton::engines::symbolic::SharedSymbolicVariable _symvar = it->second;
					if (!_symvar)
					{
						std::cout << "overlapped_reg: " << overlapped_reg << std::endl;
						const triton::ast::SharedAbstractNode reg_ast = triton_api->getRegisterAst(overlapped_reg);
						std::cout << triton_api->processSimplification(reg_ast, true) << std::endl;
						throw std::runtime_error("known reg is somehow not symbolic variable");
					}

					// load IR expression from triton symvar
					auto _it = context->m_expression_map.find(_symvar->getId());
					if (_it == context->m_expression_map.end())
						throw std::runtime_error("not in m_expression_map");

					if (op_reg.getLow() != overlapped_reg.getLow())
						throw std::runtime_error("don't know what to do if low is different");

					// extract if needed
					if (op_reg.getHigh() < overlapped_reg.getHigh())
					{
						auto extract = std::make_shared<IR::Extract>(_it->second, op_reg.getHigh(), op_reg.getLow());

						std::stringstream ss;
						extract->to_string(ss);
						triton::engines::symbolic::SharedSymbolicVariable low_reg_symvar = triton_api->symbolizeRegister(op_reg, ss.str());
						context->m_expression_map[low_reg_symvar->getId()] = extract;
						expressions.push_back(extract);
					}
					else if (op_reg.getHigh() > overlapped_reg.getHigh())
					{
						// high is unknown | low is known
						auto extend = std::make_shared<IR::Extend>(_it->second, op_reg.getSize());
						std::stringstream ss;
						extend->to_string(ss);
						triton::engines::symbolic::SharedSymbolicVariable low_reg_symvar = triton_api->symbolizeRegister(op_reg, ss.str());
						context->m_expression_map[low_reg_symvar->getId()] = extend;
						expressions.push_back(extend);
					}
					else
					{
						expressions.push_back(_it->second);
					}

					_make_imm = false;
					_save = true;
					break;
				}
			}

			// otherwise immediate
			if (_make_imm)
			{
				expressions.push_back(std::make_shared<IR::Immediate>(
					triton_api->getConcreteRegisterValue(op_reg).convert_to<triton::uint64>()));
			}
		}
		else
			throw std::runtime_error("invalid operand type");
	}

	if (!_save)
		expressions.clear();
	return expressions;
}

void ThemidaAnalyzer::check_arity_operation(triton::arch::Instruction& triton_instruction,
	const std::vector<std::shared_ptr<IR::Expression>>& operands_expressions, hdlr_ctx_ptr context)
{
	// mov special
	if (is_mov_operation(triton_instruction) && operands_expressions.size() == 1)
	{
		// mov|movsx|movzx X, IR::Expression
		auto source_expr = operands_expressions[0];
		if (triton_instruction.getType() == triton::arch::x86::ID_INS_MOVSX)
		{
			const triton::arch::Register& dest_reg = triton_instruction.operands[0].getConstRegister();
			triton::engines::symbolic::SharedSymbolicVariable symvar = triton_api->symbolizeRegister(dest_reg);
			context->m_expression_map[symvar->getId()] = std::make_shared<IR::SignExtend>(source_expr);

			// mark as known
			context->known_regs[dest_reg] = symvar;
		}
		else if (triton_instruction.getType() == triton::arch::x86::ID_INS_MOVZX)
		{
			const triton::arch::Register& dest_reg = triton_instruction.operands[0].getConstRegister();
			triton::engines::symbolic::SharedSymbolicVariable symvar = triton_api->symbolizeRegister(dest_reg);
			context->m_expression_map[symvar->getId()] = std::make_shared<IR::ZeroExtend>(source_expr);

			// mark as known
			context->known_regs[dest_reg] = symvar;
		}
		else if (triton_instruction.operands[0].getType() == triton::arch::OP_REG)
		{
			// should be mov
			const triton::arch::Register& dest_reg = triton_instruction.operands[0].getConstRegister();
			auto symvar = get_symbolic_var(this->triton_api->processSimplification(this->triton_api->getRegisterAst(dest_reg), true));
			if (!symvar)
			{
				std::cout << operands_expressions[0] << std::endl;
				throw std::runtime_error("check_arity_operation very unexpected");
			}
			context->known_regs[dest_reg] = symvar;
		}

		return;
	}

	bool unary = is_unary_operation(triton_instruction) && operands_expressions.size() == 1;
	bool binary = is_binary_operation(triton_instruction) && operands_expressions.size() == 2;
	if (!unary && !binary)
		return;

	// sym op
	auto symbolize_operand = [this](const triton::arch::OperandWrapper& op, const std::string& alias = "") -> triton::engines::symbolic::SharedSymbolicVariable
	{
		if (op.getType() == triton::arch::OP_REG)
		{
			const triton::arch::Register& _reg = op.getConstRegister();
			this->triton_api->concretizeRegister(_reg);
			return this->triton_api->symbolizeRegister(_reg, alias);
		}
		else if (op.getType() == triton::arch::OP_MEM)
		{
			const triton::arch::MemoryAccess& _mem = op.getConstMemory();
			triton_api->concretizeMemory(_mem);
			return triton_api->symbolizeMemory(_mem, alias);
		}
		else
		{
			throw std::runtime_error("invalid operand type");
		}
	};
	std::shared_ptr<IR::Expression> expr;

	const triton::arch::Register& triton_eflags = this->triton_api->registers.x86_eflags;
	const triton::arch::OperandWrapper& triton_op0 = triton_instruction.operands[0];

	if (unary)
	{
		// make unary IL expression
		auto op0_expression = operands_expressions[0];
		switch (triton_instruction.getType())
		{
			case triton::arch::x86::ID_INS_INC:
			{
				expr = std::make_shared<IR::Inc>(op0_expression);
				break;
			}
			case triton::arch::x86::ID_INS_DEC:
			{
				expr = std::make_shared<IR::Dec>(op0_expression);
				break;
			}
			case triton::arch::x86::ID_INS_NEG:
			{
				expr = std::make_shared<IR::Neg>(op0_expression);
				break;
			}
			case triton::arch::x86::ID_INS_NOT:
			{
				expr = std::make_shared<IR::Not>(op0_expression);
				break;
			}
			default:
			{
				throw std::runtime_error("unknown unary operation");
			}
		}
	}
	else if (binary)
	{
		// make binary IL expression
		auto op0_expression = operands_expressions[0];
		auto op1_expression = operands_expressions[1];
		switch (triton_instruction.getType())
		{
			case triton::arch::x86::ID_INS_ADD:
			{
				expr = std::make_shared<IR::Add>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_SUB:
			{
				expr = std::make_shared<IR::Sub>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_SHL:
			{
				expr = std::make_shared<IR::Shl>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_SHR:
			{
				expr = std::make_shared<IR::Shr>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_RCR:
			{
				expr = std::make_shared<IR::Rcr>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_RCL:
			{
				expr = std::make_shared<IR::Rcl>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_ROL:
			{
				expr = std::make_shared<IR::Rol>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_ROR:
			{
				expr = std::make_shared<IR::Ror>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_AND:
			{
				expr = std::make_shared<IR::And>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_OR:
			{
				expr = std::make_shared<IR::Or>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_XOR:
			{
				expr = std::make_shared<IR::Xor>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_IMUL:
			{
				expr = std::make_shared<IR::Imul>(op0_expression, op1_expression);
				break;
			}
			case triton::arch::x86::ID_INS_CMP:
			case triton::arch::x86::ID_INS_TEST:
			{
				// declare Temp
				auto temp = IR::Variable::create_variable(triton_eflags.getSize());

				// tvar = cmp(op0, op1)
				auto symvar_eflags = this->triton_api->symbolizeRegister(triton_eflags, temp->get_name());
				if (triton_instruction.getType() == triton::arch::x86::ID_INS_CMP)
					context->instructions.push_back(std::make_shared<IR::Assign>(temp, std::make_shared<IR::Cmp>(op0_expression, op1_expression)));
				else if (triton_instruction.getType() == triton::arch::x86::ID_INS_TEST)
					context->instructions.push_back(std::make_shared<IR::Assign>(temp, std::make_shared<IR::Test>(op0_expression, op1_expression)));
				context->m_expression_map[symvar_eflags->getId()] = temp;

				// mark as known
				context->known_regs[triton_eflags] = symvar_eflags;
				return;
			}
			default:
			{
				throw std::runtime_error("unknown binary operation");
			}
		}
	}

	// declare temp
	auto temp = IR::Variable::create_variable(triton_op0.getSize());

	// temp = UNARY_OP(op0) | BIN_OP(op0, op1)
	triton::engines::symbolic::SharedSymbolicVariable symvar = symbolize_operand(triton_op0, temp->get_name());
	context->instructions.push_back(std::make_shared<IR::Assign>(temp, expr));
	context->m_expression_map[symvar->getId()] = temp;

	// mark as known
	if (triton_op0.getType() == triton::arch::OP_REG)
	{
		// should be mov
		const triton::arch::Register& dest_reg = triton_op0.getConstRegister();
		context->known_regs[dest_reg] = symvar;
	}

	// check if flags are written
	for (const auto& pair : triton_instruction.getWrittenRegisters())
	{
		const triton::arch::Register& written_register = pair.first;
		if (this->triton_api->isFlag(written_register))
		{
			// declare Temp
			auto _tvar_eflags = IR::Variable::create_variable(triton_eflags.getSize());

			// temp = FlagOf(expr)
			auto symvar_eflags = this->triton_api->symbolizeRegister(triton_eflags, _tvar_eflags->get_name());
			context->instructions.push_back(std::make_shared<IR::Assign>(_tvar_eflags, std::make_shared<IR::Flags>(expr)));
			context->m_expression_map[symvar_eflags->getId()] = _tvar_eflags;

			// mark as known
			context->known_regs[triton_eflags] = symvar_eflags;
			break;
		}
	}
}


void ThemidaAnalyzer::run_vm_handler(AbstractStream& stream, triton::uint64 handler_address, hdlr_ctx_ptr context)
{
	std::shared_ptr<BasicBlock> basic_block;
	auto handler_it = this->m_handlers.find(handler_address);
	if (handler_it == this->m_handlers.end())
	{
		basic_block = make_cfg(stream, handler_address);
		this->m_handlers.insert(std::make_pair(handler_address, basic_block));
	}
	else
	{
		basic_block = handler_it->second;
	}

	triton::uint64 expected_return_address = 0;
	for (auto it = basic_block->instructions.begin(); it != basic_block->instructions.end();)
	{
		const std::shared_ptr<x86_instruction> xed_instruction = *it;
		const std::vector<xed_uint8_t> bytes = xed_instruction->get_bytes();
		bool mem_read = false;

		const xed_uint_t num_of_mem_ops = xed_instruction->get_number_of_memory_operands();
		for (xed_uint_t j = 0, memops = num_of_mem_ops; j < memops; j++)
		{
			if (xed_instruction->is_mem_read(j))
			{
				mem_read = true;
				break;
			}
		}

		// do stuff with triton
		triton::arch::Instruction triton_instruction;
		triton_instruction.setOpcode(&bytes[0], (triton::uint32)bytes.size());
		triton_instruction.setAddress(xed_instruction->get_addr());

		// fix ip
		triton_api->setConcreteRegisterValue(this->get_ip_register(), xed_instruction->get_addr());

		// disassembly and symbolize memory before "processing"
		triton_api->disassembly(triton_instruction);
		if (mem_read)
		{
			for (auto& operand : triton_instruction.operands)
			{
				if (operand.getType() == triton::arch::OP_MEM)
				{
					triton_api->getSymbolicEngine()->initLeaAst(operand.getMemory());
					if (this->symbolize_read_memory(operand.getConstMemory(), context))
					{
						// memory is known variable (VM_REG or deref(VM_REG))
					}
				}
			}
		}

		std::vector<std::shared_ptr<IR::Expression>> operands_expressions = this->save_expressions(triton_instruction, context);

		// buildSemantics
		triton_api->buildSemantics(triton_instruction);

		// update known_regs
		for (const auto& written_reg : triton_instruction.getWrittenRegisters())
		{
			for (const triton::arch::Register& overlapped_reg : this->get_overlapped_regs(written_reg.first))
			{
				context->known_regs.erase(overlapped_reg);
			}
		}

		// lol
		this->check_arity_operation(triton_instruction, operands_expressions, context);

		// and then check memory store
		this->storeAccess(triton_instruction, context);

		// symbolize memory if eflags is symbolized
		if (triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFD)
		{
			auto eflags_ast = this->triton_api->getRegisterAst(this->triton_api->registers.x86_eflags);
			triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(triton_api->processSimplification(eflags_ast, true));
			if (_symvar)
			{
				auto it = context->m_expression_map.find(_symvar->getId());
				if (it == context->m_expression_map.end())
				{
					// ?
					throw std::runtime_error("bluh");
				}

				triton::arch::MemoryAccess _mem(this->get_sp(), 4);
				auto _symvar = triton_api->symbolizeMemory(_mem);
				context->m_expression_map[_symvar->getId()] = it->second;
			}
		}
		else if (triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFQ)
		{
			auto eflags_ast = this->triton_api->getRegisterAst(this->triton_api->registers.x86_eflags);
			triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(triton_api->processSimplification(eflags_ast, true));
			if (_symvar)
			{
				auto it = context->m_expression_map.find(_symvar->getId());
				if (it == context->m_expression_map.end())
				{
					// ?
					throw std::runtime_error("bluh");
				}

				triton::arch::MemoryAccess _mem(this->get_sp(), 8);
				auto _symvar = triton_api->symbolizeMemory(_mem);
				context->m_expression_map[_symvar->getId()] = it->second;
			}
		}

		if (xed_instruction->get_category() != XED_CATEGORY_UNCOND_BR
			|| xed_instruction->get_branch_displacement_width() == 0)
		{
			if (print_debug)
			std::cout << "\t" << triton_instruction << "\n";
		}

		if (++it != basic_block->instructions.end())
		{
			// loop until it reaches end
			continue;
		}

		if (triton_instruction.getType() == triton::arch::x86::ID_INS_CALL)
		{
			expected_return_address = xed_instruction->get_addr() + 5;
		}
		else if (triton_instruction.getType() == triton::arch::x86::ID_INS_RET)
		{
			if (expected_return_address != 0 && this->get_ip() == expected_return_address)
			{
				basic_block = make_cfg(stream, expected_return_address);
				it = basic_block->instructions.begin();
			}
		}

		while (it == basic_block->instructions.end())
		{
			if (basic_block->next_basic_block && basic_block->target_basic_block)
			{
				// it ends with conditional branch
				if (triton_instruction.isConditionTaken())
				{
					basic_block = basic_block->target_basic_block;
				}
				else
				{
					basic_block = basic_block->next_basic_block;
				}
			}
			else if (basic_block->target_basic_block)
			{
				// it ends with jmp?
				basic_block = basic_block->target_basic_block;
			}
			else if (basic_block->next_basic_block)
			{
				// just follow :)
				basic_block = basic_block->next_basic_block;
			}
			else
			{
				// perhaps finishes?
				goto l_categorize_handler;
			}
			it = basic_block->instructions.begin();
		}
	}

l_categorize_handler:
	;
}



std::list<std::shared_ptr<IR::Instruction>> ThemidaAnalyzer::lift_vm_exit(hdlr_ctx_ptr ctx)
{
	// not the best implement but at least works
	std::stack<triton::arch::Register> modified_regs;
	std::shared_ptr<BasicBlock> basic_block = this->m_handlers[ctx->address];
	for (auto it = basic_block->instructions.begin(); it != basic_block->instructions.end();)
	{
		const std::shared_ptr<x86_instruction> xed_instruction = *it;
		const std::vector<xed_uint8_t> bytes = xed_instruction->get_bytes();

		// do stuff with triton
		triton::arch::Instruction triton_instruction;
		triton_instruction.setOpcode(&bytes[0], (triton::uint32)bytes.size());
		triton_instruction.setAddress(xed_instruction->get_addr());
		triton_api->processing(triton_instruction);

		for (const auto& pair : triton_instruction.getWrittenRegisters())
		{
			// skip IP or SP
			const triton::arch::Register& _reg = pair.first;
			if (this->is_x64())
			{
				if (_reg.getParent() == triton::arch::ID_REG_X86_RSP
					|| _reg.getParent() == triton::arch::ID_REG_X86_RIP)
				{
					continue;
				}
			}
			else
			{
				if (_reg.getParent() == triton::arch::ID_REG_X86_ESP
					|| _reg.getParent() == triton::arch::ID_REG_X86_EIP)
				{
					continue;
				}
			}

			// flag -> eflags
			if (this->triton_api->isFlag(_reg))
			{
				modified_regs.push(this->triton_api->registers.x86_eflags);
			}
			else if (_reg.getSize() == this->triton_api->getGprSize())
			{
				modified_regs.push(_reg);
			}
		}

		if (++it != basic_block->instructions.end())
		{
			// loop until it reaches end
			continue;
		}

		if (basic_block->next_basic_block && basic_block->target_basic_block)
		{
			// it ends with conditional branch
			if (triton_instruction.isConditionTaken())
			{
				basic_block = basic_block->target_basic_block;
			}
			else
			{
				basic_block = basic_block->next_basic_block;
			}
		}
		else if (basic_block->target_basic_block)
		{
			// it ends with jmp?
			basic_block = basic_block->target_basic_block;
		}
		else if (basic_block->next_basic_block)
		{
			// just follow :)
			basic_block = basic_block->next_basic_block;
		}
		else
		{
			// perhaps finishes?
			break;
		}

		it = basic_block->instructions.begin();
	}

	std::set<triton::arch::Register> _set;
	std::stack<triton::arch::Register> _final;
	while (!modified_regs.empty())
	{
		const triton::arch::Register r = modified_regs.top();
		modified_regs.pop();

		if (_set.count(r) == 0)
		{
			_set.insert(r);
			_final.push(r);
		}
	}

	std::list<std::shared_ptr<IR::Instruction>> ret;
	while (!_final.empty())
	{
		const triton::arch::Register r = _final.top();
		_final.pop();

		auto _pop = std::make_shared<IR::Pop>(std::make_shared<IR::Register>(r));
		ret.push_back(std::move(_pop));
	}
	ret.push_back(std::make_shared<IR::Ret>());

	return ret;
}
std::list<std::shared_ptr<IR::Instruction>> ThemidaAnalyzer::vmhandler_post(hdlr_ctx_ptr ctx)
{
	std::list<std::shared_ptr<IR::Instruction>> ret;

	// load all requirements after execution
	const triton::arch::Register bp_register = this->get_bp_register();
	const triton::arch::Register sp_register = this->get_sp_register();
	const triton::uint64 bytecode_addr = ctx->bytecode_addr;
	const triton::uint32 mem_size = bp_register.getSize();
	triton::arch::MemoryAccess bytecode_mem(bytecode_addr, mem_size);
	const triton::uint64 bytecode = triton_api->getConcreteMemoryValue(bytecode_mem).convert_to<triton::uint64>();
	const triton::uint64 stack = this->get_sp();

	printf("handler: %llX, bytecode: %llX\n", ctx->address, ctx->bytecode);
	printf("\tbytecode: %llX -> %llX\n", ctx->bytecode, bytecode);
	printf("\t%s: %llX -> %llX\n", sp_register.getName().c_str(), ctx->stack, stack);

	// check stack changes
	auto simplified_node = triton_api->processSimplification(triton_api->getRegisterAst(sp_register), true);
	auto symvar_sp = get_symbolic_var(simplified_node);
	const triton::uint32 stack_size = sp_register.getSize();
	const triton::sint64 stack_diff = stack - ctx->stack;
	auto ir_x86_stack = ctx->m_expression_map[ctx->symvar_stack->getId()];
	if (!symvar_sp)
	{
		// if SP is symbolic variable don't insert below
		for (triton::sint64 stack_offset = stack_diff; stack_offset < 0; stack_offset += stack_size)
		{
			triton::arch::MemoryAccess _mem(stack + stack_offset + stack_size, stack_size);
			std::shared_ptr<IR::Expression> expr;
			triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(triton_api->processSimplification(triton_api->getMemoryAst(_mem), true));
			if (_symvar)
			{
				auto _it = ctx->m_expression_map.find(_symvar->getId());
				if (_it != ctx->m_expression_map.end())
				{
					expr = _it->second;
					goto _set;
				}
			}

			// otherwise immediate
			expr = std::make_shared<IR::Immediate>(triton_api->getConcreteMemoryValue(_mem).convert_to<triton::uint64>());

		_set:
			auto ir_stack_address = std::make_shared<IR::Add>(ir_x86_stack, std::make_shared<IR::Immediate>(stack_offset));
			auto variable = std::make_shared<IR::Memory>(ir_stack_address, triton_api->registers.x86_ds, mem_size);

			// ds:[sp+stack_offset] = expr
			ret.push_back(std::make_shared<IR::Assign>(variable, expr));
		}
	}

	// insert stack changes
	if (symvar_sp)
	{
		if (symvar_sp != ctx->symvar_stack)
		{
			auto expr = ctx->triton_to_expr(symvar_sp);
			if (!expr)
				throw std::runtime_error("sp is unknown after execution");

			// sp = expr
			ret.push_back(std::make_shared<IR::Assign>(ir_x86_stack, expr));
		}
	}
	else if (simplified_node->getType() == triton::ast::BVADD_NODE
		|| simplified_node->getType() == triton::ast::BVSUB_NODE
		|| stack_diff)
	{
		auto ir_add_stack = std::make_shared<IR::Add>(ir_x86_stack, std::make_shared<IR::Immediate>(stack_diff));
		ret.push_back(std::make_shared<IR::Assign>(ir_x86_stack, ir_add_stack));
	}

	// insert branch instruction at the end
	if (ctx->jcc_detected)
	{
		const triton::uint64 target_bytecode = bytecode;
		ret.push_back(std::make_shared<IR::Jcc>(IR::Label::vip(target_bytecode)));
	}
	else if (ctx->jmp_inside_detected)
	{
		const triton::uint64 target_bytecode = bytecode;
		ret.push_back(std::make_shared<IR::Jmp>(IR::Label::vip(target_bytecode)));
	}

	return ret;
}

void ThemidaAnalyzer::vmhandler_prepare(triton::uint64 handler_address, hdlr_ctx_ptr ctx)
{
	const triton::arch::Register bp_register = this->get_bp_register();
	const triton::arch::Register sp_register = this->get_sp_register();
	const triton::arch::Register ip_register = this->get_ip_register();

	// reset
	const triton::uint64 context_address = ctx->context_addr;
	const std::vector<triton::uint8> context_bytes = triton_api->getConcreteMemoryAreaValue(context_address, THEMIDA_CONTEXT_SIZE);

	// somehow need to reset taint engine
	triton_api->clearCallbacks();
	triton_api->removeEngines();
	triton_api->initEngines();
	triton_api->addCallback(not_too_complicated);

	triton_api->concretizeAllMemory();
	triton_api->concretizeAllRegister();
	triton_api->enableTaintEngine(true);

	triton_api->setMode(triton::modes::ALIGNED_MEMORY, true);
	this->triton_api->setMode(triton::modes::AST_OPTIMIZATIONS, true);
	this->triton_api->setMode(triton::modes::CONSTANT_FOLDING, true);
	triton_api->setAstRepresentationMode(triton::ast::representations::PYTHON_REPRESENTATION);

	// at least non-zero so easy to read
	constexpr triton::uint64 c_stack_base = 0x1000;
	triton_api->setConcreteRegisterValue(sp_register, c_stack_base);
	triton_api->setConcreteRegisterValue(bp_register, context_address);
	triton_api->setConcreteMemoryAreaValue(context_address, context_bytes);

	// reset stack
	triton::uint8 zero_mem[0x1000];
	memset(zero_mem, 0, 0x1000);
	triton_api->setConcreteMemoryAreaValue(c_stack_base, zero_mem, 0x1000);

	// bp = vm context pointer
	triton::engines::symbolic::SharedSymbolicVariable symvar_context = triton_api->symbolizeRegister(bp_register, "tmd_ctx");

	// pointer to VM bytecode
	const triton::uint32 memory_size = this->is_x64() ? 8 : 4;
	triton::arch::MemoryAccess bytecode_mem(ctx->bytecode_addr, memory_size);
	triton::engines::symbolic::SharedSymbolicVariable symvar_bytecode = triton_api->symbolizeMemory(bytecode_mem, "bytecode");

	// x86 stack pointer
	triton::engines::symbolic::SharedSymbolicVariable symvar_stack = triton_api->symbolizeRegister(sp_register, "stack");

	// possible input
	auto symvar_topofstack = triton_api->symbolizeMemory(triton::arch::MemoryAccess(c_stack_base, memory_size), "topofstack");

	// init
	ctx->address = handler_address;
	ctx->stack = triton_api->getConcreteRegisterValue(sp_register).convert_to<triton::uint64>();
	ctx->bytecode = triton_api->getConcreteMemoryValue(bytecode_mem).convert_to<triton::uint64>();
	ctx->context = triton_api->getConcreteRegisterValue(bp_register).convert_to<triton::uint64>();
	ctx->symvar_stack = symvar_stack;
	ctx->symvar_bytecode = symvar_bytecode;
	ctx->symvar_context = symvar_context;
	ctx->exit_detected = false;
	ctx->jmp_inside_detected = false;
	ctx->jcc_detected = false;
	ctx->vmregs.clear();
	ctx->instructions.clear();
	ctx->m_expression_map.clear();
	ctx->known_regs.clear();
	ctx->return_address = 0;

	// no need to make ebp/rbp symbolic right?
	ctx->m_expression_map[symvar_stack->getId()] = std::make_shared<IR::Register>(this->get_sp_register());
	//context.m_expression_map[symvar_context->getId()] = std::make_shared<IR::Register>(this->get_bp_register());
	ctx->m_expression_map[symvar_bytecode->getId()] = std::make_shared<IR::Variable>("bytecode", memory_size);
	ctx->m_expression_map[symvar_topofstack->getId()] = std::make_shared<IR::Variable>("topofstack", memory_size);
}
void ThemidaAnalyzer::lift_vm_handler(AbstractStream& stream, triton::uint64 handler_address, hdlr_ctx_ptr ctx)
{
	g_stream = &stream;

	this->vmhandler_prepare(handler_address, ctx);

	// main
	try
	{
		this->run_vm_handler(stream, handler_address, ctx);
		if (ctx->exit_detected)
		{
			// jcc_out / call / jmp_out / ret
			triton::arch::MemoryAccess _mem(this->get_sp(), this->get_sp_register().getSize());
			const triton::uint64 possible_ret_address = triton_api->getConcreteMemoryValue(_mem).convert_to<triton::uint64>();
			const triton::uint64 possible_vemit_addr = this->get_ip();
			ctx->return_address = possible_ret_address;

			// collect all pushes/ret
			ctx->instructions = this->lift_vm_exit(ctx);

			//
			if (print_debug)
			std::cout << "ret handler " << std::hex << possible_vemit_addr << ", " << possible_ret_address << "\n";
			printf("ret: %016llX %016llX\n", possible_vemit_addr, possible_ret_address);
			triton_api->setConcreteRegisterValue(this->get_ip_register(), 0);

			// lazy check :|
			if (ctx->return_address == 0 && possible_vemit_addr != 0)
			{
				stream.seek(possible_vemit_addr);
				std::shared_ptr<x86_instruction> xed_instruction1 = stream.readNext();
				std::shared_ptr<x86_instruction> xed_instruction2 = stream.readNext();
				if (xed_instruction2->get_category() == XED_CATEGORY_PUSH
					|| xed_instruction2->get_category() == XED_CATEGORY_COND_BR
					|| xed_instruction2->get_category() == XED_CATEGORY_UNCOND_BR)
				{
					auto ir_instruction = std::make_shared<IR::X86Instruction>(xed_instruction1);
					printf("vemit detected, continues %llX\n", xed_instruction2->get_addr());
					ctx->instructions.push_back(ir_instruction);
					ctx->return_address = xed_instruction2->get_addr();
				}
			}
		}
		else
		{
			const triton::arch::MemoryAccess bytecode_mem(ctx->bytecode_addr, this->is_x64() ? 8 : 4);
			ctx->next_bytecode = triton_api->getConcreteMemoryValue(bytecode_mem).convert_to<triton::uint64>();

			// insert some instructions after execution
			auto insts = this->vmhandler_post(ctx);
			ctx->instructions.insert(ctx->instructions.end(), insts.begin(), insts.end());
		}
	}
	catch (const std::exception& ex)
	{
		ctx->next_handler_address = 0;
		printf("exception: %s\n", ex.what());
		throw ex;
	}

	simplify_instructions(ctx->instructions);
	ctx->next_handler_address = this->get_ip();
}


void ThemidaAnalyzer::explore_handler(AbstractStream& stream, triton::uint64 handler_address, 
	std::set<IR::Label>& visit, vm_label_t& labels, hdlr_ctx_ptr handler_ctx)
{
	auto _save_label = [this, &labels](triton::uint64 address)
	{
		const IR::Label bytecode_label = IR::Label::vip(address);
		auto it = labels.find(bytecode_label);
		if (it == labels.end())
		{
			// save bytecode, x86_ip, themida_cpu_state so we can explore later
			labels.insert(std::make_pair(bytecode_label,
				std::make_pair(this->get_ip(), this->save_cpu_state())));
		}
	};

	while (handler_address)
	{
		IR::Label vip = IR::Label::vip(handler_ctx->next_bytecode);
		if (visit.count(vip) > 0)
		{
			// done
			return;
		}
		visit.insert(vip);

		// lift vm handler with "execute_jcc = true"
		auto _cpu = this->save_cpu_state();
		handler_ctx->execute_jcc = true;
		this->lift_vm_handler(stream, handler_address, handler_ctx);
		this->m_statements[vip] = handler_ctx->instructions;

		// if jcc is detected, save cpu and re-execute handler with "execute_jcc = false"
		if (handler_ctx->jcc_detected)
		{
			_save_label(handler_ctx->next_bytecode);
			handler_ctx->execute_jcc = false;
			this->load_cpu_state(_cpu);
			this->lift_vm_handler(stream, handler_address, handler_ctx);
		}
		handler_address = handler_ctx->next_handler_address;
	}

	if (handler_ctx->return_address)
	{
		// call detected
		this->explore_entry(stream, handler_ctx->return_address, visit, labels);
	}
	else
	{
		// finished?
	}
}


void ThemidaAnalyzer::explore_entry(AbstractStream& stream, triton::uint64 vmenter_address,
	std::set<IR::Label>& visit, vm_label_t& labels)
{
	IR::Label x86_label = IR::Label::x86(vmenter_address);
	if (visit.count(x86_label) > 0)
	{
		// finish
		return;
	}
	visit.insert(x86_label);

	// analyze vm enter
	struct vmenter_objects vmenter;
	this->lift_vm_enter(stream, vmenter_address, vmenter);

	// save label
	labels[IR::Label::vip(vmenter.bytecode)] = std::make_pair(vmenter.first_handler_address, this->save_cpu_state());

	// explore vm handlers
	hdlr_ctx_ptr handler_ctx = std::make_shared<ThemidaHandlerContext>(vmenter.context_addr, vmenter.lock_addr, vmenter.bytecode_addr);
	handler_ctx->next_bytecode = vmenter.bytecode;
	this->explore_handler(stream, vmenter.first_handler_address, visit, labels, handler_ctx);
	for (auto it = labels.begin(); it != labels.end();)
	{
		const IR::Label label = it->first;
		if (!label.is_vip() || visit.count(label) > 0)
		{
			// explore only vip not visit
			++it;
			continue;
		}

		// load cpu state and explore handlers
		const std::shared_ptr<ThemidaCpuState> state = it->second.second;
		this->load_cpu_state(state);

		// like...lazy code XD
		handler_ctx->next_bytecode = label;
		this->explore_handler(stream, it->second.first, visit, labels, handler_ctx);
		it = labels.begin();
	}
}


void ThemidaAnalyzer::analyze(AbstractStream& stream, triton::uint64 vmenter_address)
{
	// <bytecode, <handler_address, cpu_state>>
	std::map<IR::Label, std::pair<triton::uint64, std::shared_ptr<ThemidaCpuState>>> labels;
	std::set<IR::Label> visit;

	// analyze vm enter
	this->explore_entry(stream, vmenter_address, visit, labels);
}



std::shared_ptr<IR::BB> make_bb(
	std::map<IR::Label, std::list<std::shared_ptr<IR::Instruction>>>& stream,
	const IR::Label& start,
	const std::set<IR::Label>& leaders, std::map<IR::Label, std::shared_ptr<IR::BB>>& basic_blocks)
{
	// return basic block if it exists
	auto _it = basic_blocks.find(start);
	if (_it != basic_blocks.end())
		return _it->second;

	// make basic block
	std::shared_ptr<IR::BB> current_basic_block = std::make_shared<IR::BB>();
	current_basic_block->label = start;
	current_basic_block->terminator = false;
	basic_blocks.insert(std::make_pair(start, current_basic_block));

	//
	auto it = stream.find(start);
	for (; it != stream.end(); ++it)
	{
		const IR::Label label = it->first;
		const std::list<std::shared_ptr<IR::Instruction>>& instructions = it->second;
		if (start != label && leaders.count(label) > 0)
		{
			// make basic block with a leader
			current_basic_block->next_basic_block = make_bb(stream, label, leaders, basic_blocks);
			return current_basic_block;
		}

		IR::Label destination_label;
		IR::handler_instructions handler_instructions;
		handler_instructions.vip = label;
		for (const std::shared_ptr<IR::Instruction>& instruction : instructions)
		{
			handler_instructions.instructions.push_back(instruction);
			switch (instruction->get_id())
			{
				case IR::instruction_id::jcc:
				{
					const std::shared_ptr<IR::Jcc> _jcc = std::dynamic_pointer_cast<IR::Jcc>(instruction);
					destination_label = _jcc->get_destination();
					break;
				}
				case IR::instruction_id::jmp:
				{
					const std::shared_ptr<IR::Jmp> _jmp = std::dynamic_pointer_cast<IR::Jmp>(instruction);
					destination_label = _jmp->get_destination();
					break;
				}
				case IR::instruction_id::ret:
				{
					current_basic_block->terminator = true;
					break;
				}
				default:
					break;
			}
		}


		current_basic_block->handler_objects.push_back(std::move(handler_instructions));
		if (destination_label.is_valid())
		{
			current_basic_block->target_basic_block = make_bb(stream, destination_label, leaders, basic_blocks);
		}
	}

	return current_basic_block;
}



void ThemidaAnalyzer::print_output()
{
	if (this->m_statements.empty())
		return;

	// identify the leaders
	//
	std::set<IR::Label> leaders;
	bool next_is_leader = false;
	std::map<IR::Label, std::list<std::shared_ptr<IR::Instruction>>> full_instructions;
	for (const auto& pair : this->m_statements)
	{
		const IR::Label label = pair.first;
		const auto& instructions = pair.second;

		// skip x86
		if (!label.is_vip())
		{
			continue;
		}
		full_instructions.insert(std::make_pair(label, instructions));

		if (next_is_leader)
		{
			leaders.insert(label);
			next_is_leader = false;
		}

		for (const std::shared_ptr<IR::Instruction>& instruction : instructions)
		{
			switch (instruction->get_id())
			{
				case IR::instruction_id::jcc:
				{
					const std::shared_ptr<IR::Jcc> _jcc = std::dynamic_pointer_cast<IR::Jcc>(instruction);
					leaders.insert(_jcc->get_destination());
					next_is_leader = true;
					break;
				}
				case IR::instruction_id::jmp:
				{
					const std::shared_ptr<IR::Jmp> _jmp = std::dynamic_pointer_cast<IR::Jmp>(instruction);
					leaders.insert(_jmp->get_destination());
					break;
				}
				case IR::instruction_id::ret:
				{
					next_is_leader = true;
					break;
				}
				default:
					break;
			}
		}
	}


	// create basic block
	//
	IR::Label first_label = full_instructions.begin()->first;
	std::map<IR::Label, std::shared_ptr<IR::BB>> basic_blocks;
	std::shared_ptr<IR::BB> bb = make_bb(full_instructions, first_label, leaders, basic_blocks);


	// apply simplification
	//
	for (int i = 0; i < 5; i++)
	{
		for (auto it = basic_blocks.rbegin(); it != basic_blocks.rend(); ++it)
		{
			// apply simplification on handler
			std::list<std::shared_ptr<IR::Instruction>> _save;
			auto& handler_objects = it->second->handler_objects;
			for (auto& handlers_instructions : handler_objects)
			{
				simplify_instructions(handlers_instructions.instructions, false);
				_save.insert(_save.end(), handlers_instructions.instructions.begin(), handlers_instructions.instructions.end());
			}

			// apply simplification on basic block
			//this->simplify_instructions(_save, true);
		}

		// need CFG to apply xchg something is wrong apparently needs to be fixed
		constexpr bool xchg_ = 1;
		if (xchg_)
		{
			for (auto bb_it = basic_blocks.rbegin(); bb_it != basic_blocks.rend(); ++bb_it)
			{
				auto& bb = bb_it->second;
				//if (!bb->terminator)
					//continue;

				simplify_xchg(bb);
			}
		}
	}

	// finally print
	//
	for (const auto& pair : basic_blocks)
	{
		std::cout << pair.first.to_string() << '\n';
		auto& bb = pair.second;
		for (const auto& handler_object : bb->handler_objects)
		{
			//std::cout << handler_object.vip.to_string() << '\n';
			for (const auto& instruction : handler_object.instructions)
			{
				std::cout << "\t" << instruction << '\n';
			}
		}
	}
}



std::shared_ptr<IR::Expression> ThemidaAnalyzer::get_vm_register(triton::uint64 offset, triton::uint32 size)
{
	auto it = this->m_vm_regs.find(offset);
	if (it != this->m_vm_regs.end())
		return it->second;

	char name[64];
	sprintf_s(name, 64, "VM_REG_%llX", offset);
	auto ret = std::make_shared<IR::Variable>(name, size);
	this->m_vm_regs.insert(std::make_pair(offset, ret));
	return ret;
}