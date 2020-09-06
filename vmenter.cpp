#include "pch.h"

#include "ThemidaAnalyzer.hpp"
#include "ThemidaIR.hpp"
#include "Statement.hpp"

#include "x86_instruction.hpp"
#include "AbstractStream.hpp"
#include "CFG.hpp"
#include "tritonhelper.hpp"
#include "BasicBlock.hpp"

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
	const triton::arch::CpuInterface* _cpu = triton_api->getCpuInstance();
	if (!_cpu)
		throw std::runtime_error("CpuInterface is nullptr");

	return _cpu->getStackPointer();
}
triton::arch::Register ThemidaAnalyzer::get_ip_register() const
{
	const triton::arch::CpuInterface* _cpu = triton_api->getCpuInstance();
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

void ThemidaAnalyzer::setConcreteMemoryAreaValue(triton::uint64 address, const std::vector<unsigned char>& values)
{
	this->triton_api->setConcreteMemoryAreaValue(address, values);
}
void ThemidaAnalyzer::setConcreteMemoryAreaValue(triton::uint64 address, const void* buf, unsigned int size)
{
	this->triton_api->setConcreteMemoryAreaValue(address, (const triton::uint8*)buf, size);
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


// VM ENTER functions
static AbstractStream* g_stream = nullptr;
static void try_reading_memory(triton::API& ctx, const triton::arch::MemoryAccess& mem)
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


void ThemidaAnalyzer::vmenter_symbolize_registers(vmenter_objects& ctx)
{
	// symbolize all registers
	auto _work = [this, &ctx](const triton::arch::Register& reg)
	{
		triton::engines::symbolic::SharedSymbolicVariable symvar = this->triton_api->symbolizeRegister(reg);
		symvar->setAlias(reg.getName());
		ctx.registers.insert(std::make_pair(symvar->getId(), std::make_shared<IR::Register>(reg)));
	};

	ctx.registers.clear();
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
void ThemidaAnalyzer::vmenter_check_lock_xchg(triton::arch::Instruction& triton_instruction, vmenter_objects& ctx)
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

	ctx.context_addr = triton_api->getConcreteRegisterValue(bp_register).convert_to<triton::uint64>();
	ctx.lock_addr = mem.getAddress();

	// load themida vm context
	const auto _old_pos = g_stream->pos();
	g_stream->seek(ctx.context_addr);

	std::vector<triton::uint8> values;
	values.resize(THEMIDA_CONTEXT_SIZE);
	if (g_stream->read(&values[0], THEMIDA_CONTEXT_SIZE) == THEMIDA_CONTEXT_SIZE)
	{
		triton_api->setConcreteMemoryAreaValue(ctx.context_addr, values);
	}
	g_stream->seek(_old_pos);

	// force JE
	this->triton_api->setConcreteRegisterValue(this->triton_api->registers.x86_zf, true);
	std::cout << "lock cmpxchg\n"
		<< '\t' << bp_register.getName() << "=0x" << std::hex << ctx.context_addr << '\n'
		<< '\t' << triton_instruction << '\n';
}
void ThemidaAnalyzer::vmenter_check_store_access(triton::arch::Instruction& triton_instruction, vmenter_objects& ctx)
{
	if (ctx.context_addr == 0)
		return;

	for (const std::pair<triton::arch::MemoryAccess, triton::ast::SharedAbstractNode>& pair : triton_instruction.getStoreAccess())
	{
		const triton::arch::MemoryAccess& mem = pair.first;
		const triton::ast::SharedAbstractNode& mem_ast = pair.second;
		const triton::uint64 address = mem.getAddress();
		if (ctx.context_addr <= address && address < (ctx.context_addr + THEMIDA_CONTEXT_SIZE))
		{
			const triton::uint64 val = triton_api->getConcreteMemoryValue(mem).convert_to<triton::uint64>();
			std::cout << "[0x" << std::hex << address << "]=0x" << std::hex << val << "\n";
			ctx.initial_data[address] = std::make_pair<>(val, mem.getSize());
		}
	}
}
void ThemidaAnalyzer::vmenter_check_eflags(triton::arch::Instruction& triton_instruction, vmenter_objects& ctx)
{
	if (ctx.eflags_written)
		return;

	// symbolize eflags if pushfd
	const triton::arch::Register& eflags_reg = triton_api->registers.x86_eflags;
	if (triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFD)
	{
		triton::arch::MemoryAccess _mem(this->get_sp(), 4);
		triton::engines::symbolic::SharedSymbolicVariable symvar_eflags = triton_api->symbolizeMemory(_mem, eflags_reg.getName());
		ctx.registers.insert(std::make_pair(symvar_eflags->getId(), std::make_shared<IR::Register>(eflags_reg)));
	}
	else if (triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFQ)
	{
		triton::arch::MemoryAccess _mem(this->get_sp(), 8);
		triton::engines::symbolic::SharedSymbolicVariable symvar_eflags = triton_api->symbolizeMemory(_mem, eflags_reg.getName());
		ctx.registers.insert(std::make_pair(symvar_eflags->getId(), std::make_shared<IR::Register>(eflags_reg)));
	}

	// check if flags are written
	for (const auto& pair : triton_instruction.getWrittenRegisters())
	{
		const triton::arch::Register& written_register = pair.first;
		if (this->triton_api->isFlag(written_register))
		{
			ctx.eflags_written = true;
			break;
		}
	}
}
void ThemidaAnalyzer::vmenter_prepare(vmenter_objects& ctx)
{
	// reset callback
	this->triton_api->clearCallbacks();
	this->triton_api->addCallback(try_reading_memory);

	// reset symbolic
	this->triton_api->concretizeAllMemory();
	this->triton_api->concretizeAllRegister();
	this->triton_api->setConcreteRegisterValue(this->get_sp_register(), ctx.original_sp);

	// reset context
	ctx.context_addr = 0;
	ctx.lock_addr = 0;
	ctx.bytecode_addr = 0;
	ctx.bytecode = 0;
	ctx.initial_data.clear();
	ctx.first_handler_address = 0;
	ctx.instructions.clear();
	ctx.eflags_written = false;
	ctx.registers.clear();

	// symbolize all registers
	this->vmenter_symbolize_registers(ctx);
}
void ThemidaAnalyzer::lift_vm_enter(AbstractStream& stream, triton::uint64 address, vmenter_objects& ctx)
{
	g_stream = &stream;

	// whatever
	ctx.original_sp = 0x1000;
	this->vmenter_prepare(ctx);

	// simulate
	std::shared_ptr<BasicBlock> basic_block = make_cfg(stream, address);
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

		this->vmenter_check_lock_xchg(triton_instruction, ctx);
		this->vmenter_check_eflags(triton_instruction, ctx);
		this->vmenter_check_store_access(triton_instruction, ctx);

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
	ctx.modified_sp = this->get_sp();
	ctx.first_handler_address = this->get_ip();
	ctx.instructions.clear();

	// check pushes
	const triton::uint32 stack_addr_size = triton_api->getGprSize();
	const triton::uint64 var_length = (ctx.original_sp - ctx.modified_sp) / stack_addr_size;
	for (triton::uint64 i = 0; i < var_length; i++)
	{
		const triton::ast::SharedAbstractNode mem_ast = triton_api->getMemoryAst(
			triton::arch::MemoryAccess(ctx.original_sp - (i * stack_addr_size) - stack_addr_size, stack_addr_size));
		const triton::ast::SharedAbstractNode simplified = triton_api->processSimplification(mem_ast, true);
		if (simplified->getType() == triton::ast::VARIABLE_NODE)
		{
			// push reg
			const triton::engines::symbolic::SharedSymbolicVariable& symvar = std::dynamic_pointer_cast<triton::ast::VariableNode>(simplified)->getSymbolicVariable();
			auto it = ctx.registers.find(symvar->getId());
			if (it == ctx.registers.end())
			{
				std::stringstream ss;
				ss << "L: " << __LINE__ << " vm-enter error " << symvar;
				throw std::runtime_error(ss.str());
			}

			std::shared_ptr<IR::Register> ir_register = it->second;
			auto _push = std::make_shared<IR::Push>(ir_register);
			ctx.instructions.push_back(std::move(_push));
		}
		else
		{
			// push immediate
			const triton::uint64 val = simplified->evaluate().convert_to<triton::uint64>();
			auto _push = std::make_shared<IR::Push>(std::make_shared<IR::Immediate>(val));
			ctx.instructions.push_back(std::move(_push));
		}
	}

	// DEBUG
	printf("0x%016llX:\n", address);
	printf("\tContextAddr: 0x%016llX\n", ctx.context_addr);
	printf("\tLockAddr: 0x%016llX\n", ctx.lock_addr);
	printf("\tHandlerAddr: 0x%016llX\n", ctx.first_handler_address);
	for (const auto& pair : ctx.initial_data)
	{
		const triton::uint64 _addr = pair.first;
		triton::arch::MemoryAccess mem(_addr, pair.second.second);
		const triton::uint64 memory_value = triton_api->getConcreteMemoryValue(mem).convert_to<triton::uint64>();
		std::cout << "[" << this->get_bp_register().getName() << "+0x" << std::hex << (_addr - ctx.context_addr) << "]=0x" << std::hex << memory_value;
		if (_addr == ctx.lock_addr)
		{
			std::cout << " <- lock";
		}
		else if (memory_value % 0x1000 == 0) // lazy check tho
		{
			std::cout << " <- module base";
		}
		else if (memory_value) // lazy check tho
		{
			ctx.bytecode = memory_value;
			ctx.bytecode_addr = _addr;
			std::cout << " <- ByteCode";
		}
		std::cout << "\n";
	}
}

// themida-cpu-state
std::shared_ptr<ThemidaCpuState> ThemidaAnalyzer::save_cpu_state()
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
	return state;
}
void ThemidaAnalyzer::load_cpu_state(const std::shared_ptr<ThemidaCpuState>& state)
{
	if (this->is_x64())
	{
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rax, state->rax);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rbx, state->rbx);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rcx, state->rcx);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rdx, state->rdx);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rsi, state->rsi);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rdi, state->rdi);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rbp, state->rbp);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rsp, state->rsp);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r8, state->r8);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r9, state->r9);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r10, state->r10);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r11, state->r11);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r12, state->r12);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r13, state->r13);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r14, state->r14);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_r15, state->r15);
	}
	else
	{
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_eax, state->rax);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_ebx, state->rbx);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_ecx, state->rcx);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_edx, state->rdx);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_esi, state->rsi);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_edi, state->rdi);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_ebp, state->rbp);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_esp, state->rsp);
	}

	const triton::uint64 context_address = state->rbp;
	triton_api->setConcreteMemoryAreaValue(context_address, state->context_data);
}