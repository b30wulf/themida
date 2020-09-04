#include "pch.h"

#include "ThemidaAnalyzer.hpp"
#include "ThemidaIR.hpp"
#include "Statement.hpp"

#include "x86_instruction.hpp"
#include "AbstractStream.hpp"
#include "CFG.hpp"
#include "tritonhelper.hpp"
#include "BasicBlock.hpp"

// hardcoded for now
#define THEMIDA_CONTEXT_SIZE	0x200
constexpr bool strict_check = false;

AbstractStream *g_stream = nullptr;

// ThemidaAnalyzer
ThemidaAnalyzer::ThemidaAnalyzer(triton::arch::architecture_e arch)
{
	this->triton_api = std::make_shared<triton::API>();
	this->triton_api->setArchitecture(arch);
	this->triton_api->setMode(triton::modes::ALIGNED_MEMORY, true);
	this->triton_api->setMode(triton::modes::AST_OPTIMIZATIONS, true);
	this->triton_api->setMode(triton::modes::CONSTANT_FOLDING, true);
	this->triton_api->setAstRepresentationMode(triton::ast::representations::PYTHON_REPRESENTATION);
}
ThemidaAnalyzer::~ThemidaAnalyzer()
{
}

bool ThemidaAnalyzer::is_x64() const
{
	const triton::arch::architecture_e architecture = this->triton_api->getArchitecture();
	switch (architecture)
	{
		case triton::arch::ARCH_X86:
			return false;

		case triton::arch::ARCH_X86_64:
			return true;

		default:
			throw std::runtime_error("invalid architecture");
	}
}

triton::arch::Register ThemidaAnalyzer::get_bp_register() const
{
	return this->is_x64() ? this->triton_api->registers.x86_rbp : this->triton_api->registers.x86_ebp;
}
triton::arch::Register ThemidaAnalyzer::get_sp_register() const
{
	const triton::arch::CpuInterface *_cpu = triton_api->getCpuInstance();
	if (!_cpu)
		throw std::runtime_error("CpuInterface is nullptr");

	return _cpu->getStackPointer();
}
triton::arch::Register ThemidaAnalyzer::get_ip_register() const
{
	const triton::arch::CpuInterface *_cpu = triton_api->getCpuInstance();
	if (!_cpu)
		throw std::runtime_error("CpuInterface is nullptr");

	return _cpu->getProgramCounter();
}

triton::uint64 ThemidaAnalyzer::get_bp() const
{
	return triton_api->getConcreteRegisterValue(this->get_bp_register()).convert_to<triton::uint64>();
}
triton::uint64 ThemidaAnalyzer::get_sp() const
{
	return triton_api->getConcreteRegisterValue(this->get_sp_register()).convert_to<triton::uint64>();
}
triton::uint64 ThemidaAnalyzer::get_ip() const
{
	return triton_api->getConcreteRegisterValue(this->get_ip_register()).convert_to<triton::uint64>();
}

std::vector<triton::arch::Register> ThemidaAnalyzer::get_overlapped_regs(const triton::arch::Register& reg) const
{
	auto it = this->m_overlapped_regs.find(reg.getId());
	if (it != this->m_overlapped_regs.end())
		return it->second;

	std::vector<triton::arch::Register> ret;
	for (const auto& pair : this->triton_api->getAllRegisters())
	{
		if (pair.second.isOverlapWith(reg))
			ret.push_back(pair.second);
	}

	this->m_overlapped_regs.insert(std::make_pair(reg.getId(), ret));
	return ret;
}

void ThemidaAnalyzer::load(AbstractStream& stream, 
	triton::uint64 module_base, triton::uint64 virtual_address, triton::uint64 virtual_size)
{
	const triton::uint64 runtime_address = module_base + virtual_address;
	void *vmp0 = malloc(virtual_size);

	stream.seek(runtime_address);
	if (stream.read(vmp0, virtual_size) != virtual_size)
		throw std::runtime_error("stream.read failed");

	triton_api->setConcreteMemoryAreaValue(runtime_address, (const triton::uint8 *)vmp0, virtual_size);
	free(vmp0);
}

// VM ENTER
void ThemidaAnalyzer::vm_enter_getConcreteMemoryValue_callback(triton::API& ctx, const triton::arch::MemoryAccess& mem)
{
	// read static memory if needed
	const triton::uint64 address = mem.getAddress();
	if (ctx.isConcreteMemoryValueDefined(address, mem.getSize()))
		return;

	const auto _old_pos = g_stream->pos();
	g_stream->seek(address);

	std::vector<triton::uint8> values;
	const triton::uint32 mem_size = mem.getSize();
	values.resize(mem_size);
	if (g_stream->read(&values[0], mem_size) == mem_size)
	{
		ctx.setConcreteMemoryAreaValue(address, values);
	}
	else
	{
		std::stringstream ss;
		ss << "Failed to read memory at 0x" << std::hex << address << "\n"
			<< "\tFile: " << __FILE__ << ", L: " << __LINE__;
		throw std::runtime_error(ss.str());
	}
	g_stream->seek(_old_pos);
}
void ThemidaAnalyzer::symbolize_registers(std::shared_ptr<ThemidaVMEnterContext> context)
{
	// symbolize all registers
	auto _work = [this, context](const triton::arch::Register& reg)
	{
		triton::engines::symbolic::SharedSymbolicVariable symvar = this->triton_api->symbolizeRegister(reg);
		symvar->setAlias(reg.getName());
		context->registers.insert(std::make_pair(symvar->getId(), std::make_shared<IR::Register>(reg)));
	};
	
	context->registers.clear();
	if (this->is_x64())
	{
		_work(triton_api->registers.x86_rax);
		_work(triton_api->registers.x86_rbx);
		_work(triton_api->registers.x86_rcx);
		_work(triton_api->registers.x86_rdx);
		_work(triton_api->registers.x86_rsi);
		_work(triton_api->registers.x86_rdi);
		_work(triton_api->registers.x86_rbp);
		//_work(triton_api->registers.x86_rsp);

		_work(triton_api->registers.x86_r8);
		_work(triton_api->registers.x86_r9);
		_work(triton_api->registers.x86_r10);
		_work(triton_api->registers.x86_r11);
		_work(triton_api->registers.x86_r12);
		_work(triton_api->registers.x86_r13);
		_work(triton_api->registers.x86_r14);
		_work(triton_api->registers.x86_r15);
	}
	else
	{
		_work(triton_api->registers.x86_eax);
		_work(triton_api->registers.x86_ebx);
		_work(triton_api->registers.x86_ecx);
		_work(triton_api->registers.x86_edx);
		_work(triton_api->registers.x86_esi);
		_work(triton_api->registers.x86_edi);
		_work(triton_api->registers.x86_ebp);
		//_work(triton_api->registers.x86_esp);
	}
}
void ThemidaAnalyzer::check_lock_xchg(triton::arch::Instruction& triton_instruction, std::shared_ptr<ThemidaVMEnterContext> context)
{
	// expected: lock cmpxchg [rbx+rbp],ecx
	if (triton_instruction.getType() != triton::arch::x86::ID_INS_CMPXCHG
		|| triton_instruction.getPrefix() != triton::arch::x86::ID_PREFIX_LOCK)
	{
		return;
	}

	const auto& loadAccess = triton_instruction.getLoadAccess();
	assert(loadAccess.size() == 1);

	// lock cmpxchg
	const triton::arch::Register bp_register = this->get_bp_register();
	const triton::arch::MemoryAccess& mem = loadAccess.begin()->first;
	const triton::ast::SharedAbstractNode& node = loadAccess.begin()->second;
	if (mem.getConstBaseRegister() != bp_register && mem.getConstIndexRegister() != bp_register)
	{
		std::stringstream ss;
		ss << "unexpected lock cmpxchg\n\t" << triton_instruction;
		throw std::runtime_error(ss.str());
	}

	context->context_address = triton_api->getConcreteRegisterValue(bp_register).convert_to<triton::uint64>();
	context->lock_address = mem.getAddress();

	// load themida vm context
	const auto _old_pos = g_stream->pos();
	g_stream->seek(context->context_address);

	std::vector<triton::uint8> values;
	values.resize(THEMIDA_CONTEXT_SIZE);
	if (g_stream->read(&values[0], THEMIDA_CONTEXT_SIZE) == THEMIDA_CONTEXT_SIZE)
	{
		triton_api->setConcreteMemoryAreaValue(context->context_address, values);
	}
	g_stream->seek(_old_pos);

	// force JE
	this->triton_api->setConcreteRegisterValue(this->triton_api->registers.x86_zf, true);
	std::cout << "lock cmpxchg\n"
		<< '\t' << bp_register.getName() << "=0x" << std::hex << context->context_address << '\n'
		<< '\t' << triton_instruction << '\n';
}
void ThemidaAnalyzer::check_store_access(triton::arch::Instruction& triton_instruction, std::shared_ptr<ThemidaVMEnterContext> context)
{
	if (context->context_address == 0)
		return;

	for (const std::pair<triton::arch::MemoryAccess, triton::ast::SharedAbstractNode>& pair : triton_instruction.getStoreAccess())
	{
		const triton::arch::MemoryAccess& mem = pair.first;
		const triton::ast::SharedAbstractNode& mem_ast = pair.second;
		const triton::uint64 address = mem.getAddress();
		if (context->context_address <= address && address < (context->context_address + THEMIDA_CONTEXT_SIZE))
		{
			const triton::uint64 val = triton_api->getConcreteMemoryValue(mem).convert_to<triton::uint64>();
			std::cout << "[0x" << std::hex << address << "]=0x" << std::hex << val << "\n";
			context->initial_data[address] = std::make_pair<>(val, mem.getSize());
		}
	}
}
void ThemidaAnalyzer::check_eflags(triton::arch::Instruction& triton_instruction, std::shared_ptr<ThemidaVMEnterContext> context)
{
	if (context->flags_written)
		return;

	// symbolize eflags if pushfd
	const triton::arch::Register& eflags_reg = triton_api->registers.x86_eflags;
	if (triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFD)
	{
		triton::arch::MemoryAccess _mem(this->get_sp(), 4);
		triton::engines::symbolic::SharedSymbolicVariable symvar_eflags = triton_api->symbolizeMemory(_mem);
		symvar_eflags->setAlias("eflags");
		context->registers.insert(std::make_pair(symvar_eflags->getId(), std::make_shared<IR::Register>(eflags_reg)));
	}
	else if (triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFQ)
	{
		triton::arch::MemoryAccess _mem(this->get_sp(), 8);
		triton::engines::symbolic::SharedSymbolicVariable symvar_eflags = triton_api->symbolizeMemory(_mem);
		symvar_eflags->setAlias("eflags");
		context->registers.insert(std::make_pair(symvar_eflags->getId(), std::make_shared<IR::Register>(eflags_reg)));
	}

	// check if flags are written
	for (const auto& pair : triton_instruction.getWrittenRegisters())
	{
		const triton::arch::Register& written_register = pair.first;
		if (this->triton_api->isFlag(written_register))
		{
			context->flags_written = true;
			break;
		}
	}
}
void ThemidaAnalyzer::prepare_vm_enter(std::shared_ptr<ThemidaVMEnterContext> context)
{
	// reset callback
	this->triton_api->clearCallbacks();
	this->triton_api->addCallback(ThemidaAnalyzer::vm_enter_getConcreteMemoryValue_callback);

	// reset symbolic
	this->triton_api->concretizeAllMemory();
	this->triton_api->concretizeAllRegister();
	this->triton_api->setConcreteRegisterValue(this->get_sp_register(), context->original_sp);

	// reset context
	context->modified_sp = 0;
	context->registers.clear();
	context->flags_written = false;
	context->context_address = 0;
	context->lock_address = 0;
	context->bytecode_address = 0;
	context->initial_data.clear();
	context->first_handler_address = 0;
	context->instructions.clear();

	// symbolize all registers
	this->symbolize_registers(context);
}
void ThemidaAnalyzer::analyze_vm_enter(AbstractStream& stream, std::shared_ptr<ThemidaVMEnterContext> ctx)
{
	g_stream = &stream;

	this->prepare_vm_enter(ctx);

	// simulate
	std::shared_ptr<BasicBlock> basic_block = make_cfg(stream, ctx->address);

	for (auto it = basic_block->instructions.begin(); it != basic_block->instructions.end();)
	{
		const std::shared_ptr<x86_instruction> xed_instruction = *it;
		const std::vector<xed_uint8_t> bytes = xed_instruction->get_bytes();

		// fix ip
		triton_api->setConcreteRegisterValue(this->get_ip_register(), xed_instruction->get_addr());

		// do stuff with triton
		triton::arch::Instruction triton_instruction;
		triton_instruction.setOpcode(&bytes[0], (triton::uint32)bytes.size());
		triton_instruction.setAddress(xed_instruction->get_addr());
		if (!triton_api->processing(triton_instruction))
		{
			throw std::runtime_error("triton processing failed");
		}

		this->check_lock_xchg(triton_instruction, ctx);
		this->check_eflags(triton_instruction, ctx);
		this->check_store_access(triton_instruction, ctx);

		if (xed_instruction->get_category() != XED_CATEGORY_UNCOND_BR
			|| xed_instruction->get_branch_displacement_width() == 0)
		{
			//std::cout << "\t" << triton_instruction << "\n";
		}

		if (++it != basic_block->instructions.end())
		{
			// loop until it reaches end
			continue;
		}

		// follow
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
				goto l_done;
			}
			it = basic_block->instructions.begin();
		}
	}


l_done:
	// check pushes
	const triton::uint32 stack_addr_size = triton_api->getGprSize();
	ctx->modified_sp = this->get_sp();
	const triton::uint64 var_length = (ctx->original_sp - ctx->modified_sp) / stack_addr_size;
	for (triton::uint64 i = 0; i < var_length; i++)
	{
		const triton::ast::SharedAbstractNode mem_ast = triton_api->getMemoryAst(
			triton::arch::MemoryAccess(ctx->original_sp - (i * stack_addr_size) - stack_addr_size, stack_addr_size));
		const triton::ast::SharedAbstractNode simplified = triton_api->processSimplification(mem_ast, true);
		if (simplified->getType() == triton::ast::VARIABLE_NODE)
		{
			// push reg
			const triton::engines::symbolic::SharedSymbolicVariable &symvar = std::dynamic_pointer_cast<triton::ast::VariableNode>(simplified)->getSymbolicVariable();
			auto it = ctx->registers.find(symvar->getId());
			if (it == ctx->registers.end())
			{
				std::stringstream ss;
				ss << "L: " << __LINE__ << " vm-enter error " << symvar;
				throw std::runtime_error(ss.str());
			}

			std::shared_ptr<IR::Register> ir_register = it->second;
			auto _push = std::make_shared<IR::Push>(ir_register);
			ctx->instructions.push_back(std::move(_push));
		}
		else
		{
			// push immediate
			const triton::uint64 val = simplified->evaluate().convert_to<triton::uint64>();
			auto _push = std::make_shared<IR::Push>(std::make_shared<IR::Immediate>(val));
			ctx->instructions.push_back(std::move(_push));
		}
	}

	// all you need to save:
	// EBP(context_address), first_handler_address, pushes(instructions)
	//ctx->context_address = this->get_bp();
	ctx->first_handler_address = this->get_ip();

	// DEBUG
	printf("ContextAddr: 0x%016llX\n", ctx->context_address);
	printf("LockAddr: 0x%016llX\n", ctx->lock_address);
	printf("HandlerAddr: 0x%016llX\n", this->get_ip());
	for (const auto& pair : ctx->initial_data)
	{
		const triton::uint64 _addr = pair.first;
		triton::arch::MemoryAccess mem(_addr, pair.second.second);
		const triton::uint64 memory_value = triton_api->getConcreteMemoryValue(mem).convert_to<triton::uint64>();
		std::cout << "[" << this->get_bp_register().getName() << "+0x" << std::hex << (_addr - ctx->context_address) << "]=0x" << std::hex << memory_value;
		if (_addr == ctx->lock_address)
		{
			std::cout << " <- lock";
		}
		else if (memory_value % 0x1000 == 0) // lazy check tho
		{
			std::cout << " <- module base";
		}
		else if (memory_value) // lazy check tho
		{
			ctx->bytecode_address = _addr;
			std::cout << " <- ByteCode";
		}
		std::cout << "\n";
	}

	this->m_statements[IR::Label::x86(ctx->address)] = ctx->instructions;
}

// VM HANDLER
bool is_lea_tainted(triton::API& ctx, const triton::arch::MemoryAccess& mem)
{
	constexpr bool check_segment_register = false;
	return ctx.isRegisterTainted(mem.getConstBaseRegister())
		|| ctx.isRegisterTainted(mem.getConstIndexRegister())
		|| (check_segment_register && ctx.isRegisterTainted(mem.getConstSegmentRegister()));
}
bool is_bytecode_address(const triton::ast::SharedAbstractNode &lea_ast, std::shared_ptr<ThemidaHandlerContext> context)
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
bool is_stack_address(const triton::ast::SharedAbstractNode &lea_ast, std::shared_ptr<ThemidaHandlerContext> context)
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
bool is_context_address(const triton::ast::SharedAbstractNode &lea_ast, std::shared_ptr<ThemidaHandlerContext> context)
{
	// just check runtime address for now
	const triton::uint64 runtime_address = lea_ast->evaluate().convert_to<triton::uint64>();
	return context->context <= runtime_address && runtime_address < (context->context + THEMIDA_CONTEXT_SIZE);
}
bool is_deref_vmvar(const triton::ast::SharedAbstractNode &lea_ast, std::shared_ptr<ThemidaHandlerContext> context)
{
	if (lea_ast->getType() != triton::ast::VARIABLE_NODE)
		return false;

	const triton::engines::symbolic::SharedSymbolicVariable &symvar =
		std::dynamic_pointer_cast<triton::ast::VariableNode>(lea_ast)->getSymbolicVariable();
	return context->vmregs.find(symvar->getId()) != context->vmregs.end();
}

// themida-cpu-state
std::shared_ptr<ThemidaCpuState> ThemidaAnalyzer::save_cpu_state(std::shared_ptr<ThemidaVMEnterContext> ctx)
{
	std::shared_ptr<ThemidaCpuState> state = std::make_shared<ThemidaCpuState>();
	if (this->is_x64())
	{
		state->rax = triton_api->getConcreteRegisterValue(triton_api->registers.x86_rax).convert_to<triton::uint64>();
		state->rbx = triton_api->getConcreteRegisterValue(triton_api->registers.x86_rbx).convert_to<triton::uint64>();
		state->rcx = triton_api->getConcreteRegisterValue(triton_api->registers.x86_rcx).convert_to<triton::uint64>();
		state->rdx = triton_api->getConcreteRegisterValue(triton_api->registers.x86_rdx).convert_to<triton::uint64>();
		state->rsi = triton_api->getConcreteRegisterValue(triton_api->registers.x86_rsi).convert_to<triton::uint64>();
		state->rdi = triton_api->getConcreteRegisterValue(triton_api->registers.x86_rdi).convert_to<triton::uint64>();
		state->rbp = triton_api->getConcreteRegisterValue(triton_api->registers.x86_rbp).convert_to<triton::uint64>();
		state->rsp = triton_api->getConcreteRegisterValue(triton_api->registers.x86_rsp).convert_to<triton::uint64>();
		state->r8 = triton_api->getConcreteRegisterValue(triton_api->registers.x86_r8).convert_to<triton::uint64>();
		state->r9 = triton_api->getConcreteRegisterValue(triton_api->registers.x86_r9).convert_to<triton::uint64>();
		state->r10 = triton_api->getConcreteRegisterValue(triton_api->registers.x86_r10).convert_to<triton::uint64>();
		state->r11 = triton_api->getConcreteRegisterValue(triton_api->registers.x86_r11).convert_to<triton::uint64>();
		state->r12 = triton_api->getConcreteRegisterValue(triton_api->registers.x86_r12).convert_to<triton::uint64>();
		state->r13 = triton_api->getConcreteRegisterValue(triton_api->registers.x86_r13).convert_to<triton::uint64>();
		state->r14 = triton_api->getConcreteRegisterValue(triton_api->registers.x86_r14).convert_to<triton::uint64>();
		state->r15 = triton_api->getConcreteRegisterValue(triton_api->registers.x86_r15).convert_to<triton::uint64>();
	}
	else
	{
		// u only need rbp tho
		state->rax = triton_api->getConcreteRegisterValue(triton_api->registers.x86_eax).convert_to<triton::uint64>();
		state->rbx = triton_api->getConcreteRegisterValue(triton_api->registers.x86_ebx).convert_to<triton::uint64>();
		state->rcx = triton_api->getConcreteRegisterValue(triton_api->registers.x86_ecx).convert_to<triton::uint64>();
		state->rdx = triton_api->getConcreteRegisterValue(triton_api->registers.x86_edx).convert_to<triton::uint64>();
		state->rsi = triton_api->getConcreteRegisterValue(triton_api->registers.x86_esi).convert_to<triton::uint64>();
		state->rdi = triton_api->getConcreteRegisterValue(triton_api->registers.x86_edi).convert_to<triton::uint64>();
		state->rbp = triton_api->getConcreteRegisterValue(triton_api->registers.x86_ebp).convert_to<triton::uint64>();
		state->rsp = triton_api->getConcreteRegisterValue(triton_api->registers.x86_esp).convert_to<triton::uint64>();
	}
	const triton::uint64 context_address = state->rbp;
	state->context_data = triton_api->getConcreteMemoryAreaValue(context_address, THEMIDA_CONTEXT_SIZE);
	state->vmenter_ctx = ctx;
	return state;
}
void ThemidaAnalyzer::load_cpu_state(const std::shared_ptr<ThemidaCpuState> & context, std::shared_ptr<ThemidaHandlerContext> ctx)
{
	if (this->is_x64())
	{
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rax, context->rax);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rbx, context->rbx);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rcx, context->rcx);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rdx, context->rdx);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rsi, context->rsi);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rdi, context->rdi);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rbp, context->rbp);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rsp, context->rsp);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r8, context->r8);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r9, context->r9);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r10, context->r10);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r11, context->r11);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r12, context->r12);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r13, context->r13);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r14, context->r14);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r15, context->r15);
	}
	else
	{
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_eax, context->rax);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_ebx, context->rbx);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_ecx, context->rcx);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_edx, context->rdx);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_esi, context->rsi);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_edi, context->rdi);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_ebp, context->rbp);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_esp, context->rsp);
	}

	const triton::uint64 context_address = context->rbp;
	triton_api->setConcreteMemoryAreaValue(context_address, context->context_data);
	ctx->vmenter_ctx = context->vmenter_ctx;
}


// analysis
void try_reading_memory(triton::API& ctx, const triton::arch::MemoryAccess& mem)
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
			std::cout << "Load [EBP+0x" << std::hex << offset << "](VM_REGISTER)\n";

			// temp = VM_REG
			auto _source = this->get_vm_register(offset, mem.getSize());
			context->m_statements.push_back(std::make_shared<IR::Assign>(temp, _source));
			context->m_expression_map[symvar_vmreg->getId()] = temp;

			return true;
		}
		else
		{
			std::cout << "Load [EBP+0x" << std::hex << offset << "](STATIC)\n";
		}

		const triton::uint64 vm_jcc_offset 
		//= 0x111; // fish64 white
		= 0xbb; // fish32 white
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
	else if (is_deref_vmvar(lea_ast, context))
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
		std::cout << "Deref(" << lea_ast << "," << segment_register.getName() << ")\n";

		// temp = deref(expr)
		std::shared_ptr<IR::Expression> _memory = std::make_shared<IR::Memory>(expr, segment_register, mem.getSize());
		context->m_statements.push_back(std::make_shared<IR::Assign>(temp, _memory));
		context->m_expression_map[symvar->getId()] = temp;

		return true;
	}
	else
	{
		try_reading_memory(*triton_api, mem);
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
		if (address == context->vmenter_ctx->lock_address)
		{
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

		if (address == context->vmenter_ctx->bytecode_address)
		{
			// move themida-instruction-pointer
			const triton::uint64 bytecode = triton_api->getConcreteMemoryValue(mem).convert_to<triton::uint64>();
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
				std::cout << "Store VM_REG [EBP+0x" << std::hex << offset << "]\n";

				// create IR (VM_REG = mem_ast)
				auto source_node = triton_api->processSimplification(mem_ast, true);
				if (source_node->getType() == triton::ast::BV_NODE)
				{
					// VM_REG_X = immediate
					std::shared_ptr<IR::Expression> v1 = this->get_vm_register(offset, mem.getSize()); // dont inc count here
					std::shared_ptr<IR::Instruction> _assign = std::make_shared<IR::Assign>(v1, std::make_shared<IR::Immediate>(
						source_node->evaluate().convert_to<triton::uint64>()));
					context->m_statements.push_back(_assign);
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
						context->m_statements.push_back(std::make_shared<IR::Assign>(v1, expr));
					}
					else if (symvar->getAlias().find("topofstack") != std::string::npos) // nice try ^_^
					{
						// VM_REG_X = topofstack
						std::shared_ptr<IR::Expression> expr = std::make_shared<IR::Variable>("TopOfStack", mem.getSize());
						context->m_statements.push_back(std::make_shared<IR::Assign>(v1, expr));
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
					context->m_statements.push_back(std::make_shared<IR::Assign>(v1, expr));
					context->runtime_memory[v1] = address;
				}
				else
				{
					throw std::runtime_error("what do you mean 2");
				}
			}
			else
			{
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
	const std::vector<std::shared_ptr<IR::Expression>>& operands_expressions, std::shared_ptr<ThemidaHandlerContext> context)
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
					context->m_statements.push_back(std::make_shared<IR::Assign>(temp, std::make_shared<IR::Cmp>(op0_expression, op1_expression)));
				else if (triton_instruction.getType() == triton::arch::x86::ID_INS_TEST)
					context->m_statements.push_back(std::make_shared<IR::Assign>(temp, std::make_shared<IR::Test>(op0_expression, op1_expression)));
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
	context->m_statements.push_back(std::make_shared<IR::Assign>(temp, expr));
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
			context->m_statements.push_back(std::make_shared<IR::Assign>(_tvar_eflags, std::make_shared<IR::Flags>(expr)));
			context->m_expression_map[symvar_eflags->getId()] = _tvar_eflags;

			// mark as known
			context->known_regs[triton_eflags] = symvar_eflags;
			break;
		}
	}
}


void ThemidaAnalyzer::analyze_vm_handler_sub(AbstractStream& stream, triton::uint64 handler_address, std::shared_ptr<ThemidaHandlerContext> context)
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







// just make it bv if node gets too complicated
//
triton::ast::SharedAbstractNode nosimpl(triton::API& api, const triton::ast::SharedAbstractNode& node)
{
	if (node->getLevel() >= 30 && node->isSymbolized())
	{
		return api.getAstContext()->bv(node->evaluate(), node->getBitvectorSize());
	}
	return node;
}

void ThemidaAnalyzer::analyze_vm_handler(AbstractStream& stream, triton::uint64 handler_address, std::shared_ptr<ThemidaHandlerContext> ctx)
{
	std::shared_ptr<ThemidaCpuState> vm_cpu_state = this->save_cpu_state(ctx->vmenter_ctx);

	const triton::arch::Register bp_register = this->get_bp_register();
	const triton::arch::Register sp_register = this->get_sp_register();
	const triton::arch::Register ip_register = this->get_ip_register();

	// reset
	//const triton::uint64 context_address = triton_api->getConcreteRegisterValue(bp_register).convert_to<triton::uint64>();
	const triton::uint64 context_address = ctx->vmenter_ctx->context_address;
	const std::vector<triton::uint8> context_bytes = triton_api->getConcreteMemoryAreaValue(context_address, THEMIDA_CONTEXT_SIZE);

	// somehow need to reset taint engine
	triton_api->clearCallbacks();
	triton_api->removeEngines();
	triton_api->initEngines();
	triton_api->addCallback(nosimpl);

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
	triton::engines::symbolic::SharedSymbolicVariable symvar_context = triton_api->symbolizeRegister(bp_register);

	// pointer to VM bytecode
	const triton::uint32 memory_size = this->is_x64() ? 8 : 4;
	triton::arch::MemoryAccess bytecode_mem(ctx->vmenter_ctx->bytecode_address, memory_size);
	triton::engines::symbolic::SharedSymbolicVariable symvar_bytecode = triton_api->symbolizeMemory(bytecode_mem);

	// x86 stack pointer
	triton::engines::symbolic::SharedSymbolicVariable symvar_stack = triton_api->symbolizeRegister(sp_register);

	// sym
	auto symvar_topofstack = triton_api->symbolizeMemory(triton::arch::MemoryAccess(c_stack_base, memory_size));
	symvar_topofstack->setAlias("topofstack"); // test
	symvar_context->setAlias("context");
	symvar_bytecode->setAlias("bytecode");
	symvar_stack->setAlias("stack");

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
	ctx->m_statements.clear();
	ctx->m_expression_map.clear();
	ctx->known_regs.clear();
	ctx->return_address = 0;

	// no need to make ebp/rbp symbolic right?
	ctx->m_expression_map[symvar_stack->getId()] = std::make_shared<IR::Register>(this->get_sp_register());
	//context.m_expression_map[symvar_context->getId()] = std::make_shared<IR::Register>(this->get_bp_register());
	ctx->m_expression_map[symvar_bytecode->getId()] = std::make_shared<IR::Variable>("bytecode", memory_size);
	ctx->m_expression_map[symvar_topofstack->getId()] = std::make_shared<IR::Variable>("topofstack", memory_size);

	if (ctx->execute_jcc)
	{
		if (this->m_statements.find(IR::Label::vip(ctx->bytecode)) != this->m_statements.end())
		{
			ctx->next_handler_address = 0;
			ctx->return_address = 0;
			return;
		}
	}

	// main
	try
	{
		this->analyze_vm_handler_sub(stream, handler_address, ctx);
		if (ctx->exit_detected)
		{
			triton::arch::MemoryAccess _mem(this->get_sp(), memory_size);
			const triton::uint64 possible_ret_address = triton_api->getConcreteMemoryValue(_mem).convert_to<triton::uint64>();
			const triton::uint64 possible_vemit = this->get_ip();
			ctx->return_address = possible_ret_address;

			std::cout << "ret handler " << std::hex << this->get_ip() << ", " << possible_ret_address << "\n";
			printf("%016llX %016llX\n", this->get_ip(), possible_ret_address);
			triton_api->setConcreteRegisterValue(this->get_ip_register(), 0);
			this->analyze_vm_exit(ctx);

			// lazy check :|
			if (ctx->return_address == 0 && possible_vemit != 0)
			{
				stream.seek(possible_vemit);
				std::shared_ptr<x86_instruction> xed_instruction1 = stream.readNext();
				std::shared_ptr<x86_instruction> xed_instruction2 = stream.readNext();
				if (xed_instruction2->get_category() == XED_CATEGORY_PUSH
					|| xed_instruction2->get_category() == XED_CATEGORY_COND_BR
					|| xed_instruction2->get_category() == XED_CATEGORY_UNCOND_BR)
				{
					auto ir_instruction = std::make_shared<IR::X86Instruction>(xed_instruction1);
					printf("vemit detected, continues %llX\n", xed_instruction2->get_addr());
					ctx->m_statements.push_back(ir_instruction);
					ctx->return_address = xed_instruction2->get_addr();
				}
			}
		}
		this->categorize_handler(ctx);
		if (ctx->jcc_detected && ctx->execute_jcc)
		{
			const triton::arch::MemoryAccess bytecode_mem(ctx->vmenter_ctx->bytecode_address, memory_size);
			const triton::uint64 bytecode_destination = triton_api->getConcreteMemoryValue(bytecode_mem).convert_to<triton::uint64>();
			const IR::Label bytecode_label = IR::Label::vip(bytecode_destination);
			auto it = this->m_destinations.find(bytecode_label);
			if (it == this->m_destinations.end())
			{
				// save bytecode, x86_ip, themida_cpu_state so we can explore later
				this->m_destinations.insert(std::make_pair(bytecode_label,
					std::make_pair(this->get_ip(), this->save_cpu_state(ctx->vmenter_ctx))));
			}

			// re-execute handler with "execute_jcc = false"
			std::shared_ptr<ThemidaHandlerContext> handler_ctx = std::make_shared<ThemidaHandlerContext>();
			handler_ctx->execute_jcc = false;
			this->load_cpu_state(vm_cpu_state, handler_ctx);
			this->analyze_vm_handler(stream, handler_address, handler_ctx);
		}
		else if (ctx->jmp_inside_detected)
		{
			// save bytecode, x86_ip, themida_cpu_state so we can explore later
			triton::arch::MemoryAccess bytecode_mem(ctx->vmenter_ctx->bytecode_address, memory_size);
			const triton::uint64 target_bytecode = triton_api->getConcreteMemoryValue(bytecode_mem).convert_to<triton::uint64>();
			const IR::Label vip = IR::Label::vip(target_bytecode);
			auto it = this->m_destinations.find(vip);
			if (it == this->m_destinations.end())
			{
				this->m_destinations.insert(std::make_pair(vip,
					std::make_pair(this->get_ip(), this->save_cpu_state(ctx->vmenter_ctx))));
			}
		}
	}
	catch (const std::exception& ex)
	{
		this->categorize_handler(ctx);
		ctx->next_handler_address = 0;
		printf("exception: %s\n", ex.what());
		throw ex;
	}

	ctx->next_handler_address = this->get_ip();
}
void ThemidaAnalyzer::analyze_vm_exit(std::shared_ptr<ThemidaHandlerContext> ctx)
{
	// not the best impl
	std::stack<triton::arch::Register> modified_registers;
	const triton::arch::Register sp_register = this->get_sp_register();
	const triton::uint64 previous_stack = triton_api->getConcreteRegisterValue(sp_register).convert_to<triton::uint64>();

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

			// flags -> eflags
			if (this->triton_api->isFlag(_reg))
			{
				modified_registers.push(this->triton_api->registers.x86_eflags);
			}
			else if (this->is_x64())
			{
				if (_reg.getSize() == 8)
				{
					modified_registers.push(_reg);
				}
			}
			else if (_reg.getSize() == 4)
			{
				modified_registers.push(_reg);
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
	while (!modified_registers.empty())
	{
		triton::arch::Register r = modified_registers.top();
		modified_registers.pop();

		if (_set.count(r) == 0)
		{
			_set.insert(r);
			_final.push(r);
		}
	}

	while (!_final.empty())
	{
		triton::arch::Register r = _final.top();
		_final.pop();

		auto _pop = std::make_shared<IR::Pop>(std::make_shared<IR::Register>(r));
		ctx->m_statements.push_back(std::move(_pop));
	}
	ctx->m_statements.push_back(std::make_shared<IR::Ret>());
}
void ThemidaAnalyzer::categorize_handler(std::shared_ptr<ThemidaHandlerContext> context)
{
	const triton::arch::Register bp_register = this->get_bp_register();
	const triton::arch::Register sp_register = this->get_sp_register();

	const triton::uint32 mem_size = bp_register.getSize();
	triton::arch::MemoryAccess bytecode_mem(context->vmenter_ctx->bytecode_address, mem_size);
	const triton::uint64 bytecode = triton_api->getConcreteMemoryValue(bytecode_mem).convert_to<triton::uint64>();
	const triton::uint64 stack = this->get_sp();

	std::cout << "handlers outputs:\n\n\n" << "\n";
	printf("handler: %llX, bytecode: %llX\n", context->address, bytecode);
	//printf("\tstack: %llX -> %llX\n", context->stack, stack);

	// check stack changes
	// 	context.m_expression_map[symvar_stack->getId()] = std::make_shared<IR::Register>(this->get_sp_register());
	const int stack_addr_width = this->triton_api->getGprSize();
	auto ir_stack = std::make_shared<IR::Variable>("stack", stack_addr_width);
	for (triton::sint64 stack_offset = stack < context->stack ? -stack_addr_width : 0; stack_offset < 0; stack_offset += stack_addr_width)
	{
		triton::arch::MemoryAccess _mem(stack + stack_offset + stack_addr_width, stack_addr_width);
		std::shared_ptr<IR::Expression> expr;
		triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(triton_api->processSimplification(triton_api->getMemoryAst(_mem), true));
		if (_symvar)
		{
			auto _it = context->m_expression_map.find(_symvar->getId());
			if (_it != context->m_expression_map.end())
			{
				expr = _it->second;
				goto _set;
			}
		}

		// otherwise immediate
		expr = std::make_shared<IR::Immediate>(triton_api->getConcreteMemoryValue(_mem).convert_to<triton::uint64>());

	_set:
		auto ir_stack_address = std::make_shared<IR::Add>(ir_stack, std::make_shared<IR::Immediate>(stack_offset));
		auto variable = std::make_shared<IR::Memory>(ir_stack_address, triton_api->registers.x86_ds, mem_size);
		auto assign = std::make_shared<IR::Assign>(variable, expr);
		context->m_statements.push_back(assign);
	}

	// branch
	if (context->jcc_detected && context->execute_jcc)
	{
		triton::arch::MemoryAccess bytecode_mem(context->vmenter_ctx->bytecode_address, mem_size);
		const triton::uint64 target_bytecode = triton_api->getConcreteMemoryValue(bytecode_mem).convert_to<triton::uint64>();
		context->m_statements.push_back(std::make_shared<IR::Jcc>(IR::Label::vip(target_bytecode)));
	}
	else if (context->jmp_inside_detected)
	{
		triton::arch::MemoryAccess bytecode_mem(context->vmenter_ctx->bytecode_address, mem_size);
		const triton::uint64 target_bytecode = triton_api->getConcreteMemoryValue(bytecode_mem).convert_to<triton::uint64>();
		context->m_statements.push_back(std::make_shared<IR::Jmp>(IR::Label::vip(target_bytecode)));
	}

	// only log when execute_jcc is flagged
	if (context->execute_jcc)
	{
		this->simplify_statements(context);
		this->m_statements[IR::Label::vip(context->bytecode)] = context->m_statements;
	}
}


void ThemidaAnalyzer::analyze(AbstractStream& stream, triton::uint64 vmenter_address)
{
	// clear
	//this->m_labels.clear();
	//this->m_vm_regs.clear();
	//this->m_statements.clear();

	while (vmenter_address)
	{
		// analyze vm enter
		std::shared_ptr<ThemidaVMEnterContext> ctx = std::make_shared<ThemidaVMEnterContext>(vmenter_address, 0x0019F8B0);
		this->analyze_vm_enter(stream, ctx);
		triton::uint64 handler_address = ctx->first_handler_address;

		// explore vm handlers
		std::shared_ptr<ThemidaHandlerContext> handler_ctx = std::make_shared<ThemidaHandlerContext>();
		handler_ctx->vmenter_ctx = ctx;
		while (handler_address)
		{
			handler_ctx->execute_jcc = true;
			this->analyze_vm_handler(stream, handler_address, handler_ctx);
			handler_address = handler_ctx->next_handler_address;
		}

		if (handler_ctx->return_address)
		{
			// call detected
			vmenter_address = handler_ctx->return_address;
		}
		else
		{
			// finished?
			vmenter_address = 0;
		}
	}

	auto labels = this->m_destinations;
	do
	{
		labels = this->m_destinations;
		for (const auto &pair : labels)
		{
			const IR::Label label = pair.first;
			if (!label.is_vip())
			{
				// explore only vip
				continue;
			}

			triton::uint64 handler_address = pair.second.first;
			const std::shared_ptr<ThemidaCpuState> &state = pair.second.second;

			std::shared_ptr<ThemidaHandlerContext> handler_ctx = std::make_shared<ThemidaHandlerContext>();
			this->load_cpu_state(state, handler_ctx);
			while (handler_address)
			{
				handler_ctx->execute_jcc = true;
				this->analyze_vm_handler(stream, handler_address, handler_ctx);
				handler_address = handler_ctx->next_handler_address;
			}

			if (handler_ctx->return_address)
			{
				this->analyze(stream, handler_ctx->return_address);
			}
			else
			{
				;
			}
		}
	} while (labels.size() != this->m_destinations.size());
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
				this->simplify_instructions(handlers_instructions.instructions, false);
				_save.insert(_save.end(), handlers_instructions.instructions.begin(), handlers_instructions.instructions.end());
			}

			// apply simplification on basic block
			//this->simplify_instructions(_save, true);
		}

		// need CFG
		constexpr bool xchg_ = 1;
		if (xchg_)
		{
			for (auto bb_it = basic_blocks.rbegin(); bb_it != basic_blocks.rend(); ++bb_it)
			{
				auto& bb = bb_it->second;
				//if (!bb->terminator)
					//continue;

				std::map<IR::exprptr, IR::exprptr> xchgmap;
				for (auto ho_it = bb->handler_objects.begin(); ho_it != bb->handler_objects.end(); ho_it++)
				{
					auto& handlers_instructions = ho_it->instructions;
					std::function<IR::exprptr(IR::exprptr)> apply_xchg = [&apply_xchg, &xchgmap](IR::exprptr expression) -> std::shared_ptr<IR::Expression>
					{
						switch (expression->get_type())
						{
							case IR::expr_register:
							case IR::expr_variable:
							{
								auto it = xchgmap.find(expression);
								return it != xchgmap.end() ? it->second : expression;
							}
							case IR::expr_memory:
							{
								// maybe [base+index*scale+disp]?
								std::shared_ptr<IR::Memory> mem = std::dynamic_pointer_cast<IR::Memory>(expression);
								mem->set_expression(apply_xchg(mem->get_expression()));
								break;
							}
							case IR::expr_immediate:
							{
								break;
							}
							case IR::expr_unary_operation:
							{
								expression->set_operand(0, apply_xchg(expression->get_operand(0)));
								break;
							}
							case IR::expr_binary_operation:
							{
								expression->set_operand(0, apply_xchg(expression->get_operand(0)));
								expression->set_operand(1, apply_xchg(expression->get_operand(1)));
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
					for (auto hi_it = handlers_instructions.begin(); hi_it != handlers_instructions.end();)
					{
						std::shared_ptr<IR::Instruction> instruction = *hi_it;
						switch (instruction->get_id())
						{
							case IR::instruction_id::assign:
							{
								instruction->set_lhs(apply_xchg(instruction->get_lhs()));
								instruction->set_rhs(apply_xchg(instruction->get_rhs()));
								break;
							}
							case IR::instruction_id::xchg:
							{
								const auto lhs = instruction->get_lhs();
								const auto rhs = instruction->get_rhs();

								// read left
								const auto xchged_lhs = apply_xchg(lhs);
								const auto xchged_rhs = apply_xchg(rhs);

								xchgmap[lhs] = xchged_rhs;
								xchgmap[rhs] = xchged_lhs;
								hi_it = handlers_instructions.erase(hi_it);
								continue;
							}
							case IR::instruction_id::push:
							case IR::instruction_id::pop:
							{
								instruction->set_expression(apply_xchg(instruction->get_expression()));
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
							}
						}

						++hi_it;
					}
				}
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
			std::cout << handler_object.vip.to_string() << '\n';
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


//
void ThemidaAnalyzer::simplify_instructions(std::list<std::shared_ptr<IR::Instruction>>& instructions, bool basic_block)
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
	return;
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

				instruction->set_rhs(propagation(instruction->get_rhs()));
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
void ThemidaAnalyzer::simplify_statements(std::shared_ptr<ThemidaHandlerContext> context)
{
	for (int i = 0; i < 5; i++)
	{
		this->simplify_instructions(context->m_statements);
	}

	if (context->exit_detected)
		return;

	// check memory
	auto& instructions = context->m_statements;
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
				if (lhs->get_type() == IR::expression_type::expr_memory)
				{
					auto _it = context->runtime_memory.find(lhs);
					if (_it != context->runtime_memory.end())
					{
						if (_it->second < this->get_sp())
						{
							// invalid
							instructions.erase(--(it.base()));
							continue;
						}
					}
				}
				break;
			}
			case IR::instruction_id::xchg:
			{
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
}

void ThemidaAnalyzer::setConcreteMemoryAreaValue(triton::uint64 address, const std::vector<unsigned char>& d)
{
	this->triton_api->setConcreteMemoryAreaValue(address, d);
}
void ThemidaAnalyzer::setConcreteMemoryAreaValue(triton::uint64 address, const void* buf, unsigned int size)
{
	this->triton_api->setConcreteMemoryAreaValue(address, (const triton::uint8*)buf, size);
}