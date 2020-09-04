#pragma once

#include "AbstractStream.hpp"
#include "Label.hpp"

// forward declaration
struct BasicBlock;
namespace IR
{
	class Expression;
	class Register;
	class Instruction;
}

// ctx
struct ThemidaVMEnterContext
{
	// save
	struct
	{
		// vm enter address must be valid
		triton::uint64 address;

		// store original stack pointer before enter themida VM so we can check stack later
		triton::uint64 original_sp;

		// sp after execution
		triton::uint64 modified_sp;
	};

	// <triton_symid, expression>
	std::map<triton::usize, std::shared_ptr<IR::Register>> registers;

	// gather information while analyzing
	struct
	{
		// for pushfd/pushfq
		bool flags_written;
		triton::uint64 context_address, lock_address, bytecode_address;

		// <runtime_address, <value, size>>
		std::map<triton::uint64, std::pair<triton::uint64, triton::uint32>> initial_data;

		// handler address that will be executed after vmenter
		triton::uint64 first_handler_address;

		// VMEnter instructions -> IR
		std::list<std::shared_ptr<IR::Instruction>> instructions;
	};

	ThemidaVMEnterContext(triton::uint64 addr = 0, triton::uint64 sp = 0x0019F8B0)
	{
		this->address = addr;
		// set x86 stack pointer (at least set to non-zero so easy to read)
		this->original_sp = sp;
		this->modified_sp = 0;
		this->flags_written = 0;
		this->context_address = 0;
		this->lock_address = 0;
		this->bytecode_address = 0;
		this->first_handler_address = 0;
	}
};
struct ThemidaHandlerContext
{
	// information gathered from vm enter
	std::shared_ptr<ThemidaVMEnterContext> vmenter_ctx;

	// before start
	triton::uint64 address;
	triton::uint64 stack, bytecode, context;
	triton::engines::symbolic::SharedSymbolicVariable symvar_stack, symvar_bytecode, symvar_context
		;
	bool execute_jcc;

	// <triton_symid, symvar>
	std::map<triton::usize, triton::engines::symbolic::SharedSymbolicVariable> vmregs;

	// ret
	bool exit_detected, jmp_inside_detected, jcc_detected;

	// expressions
	std::list<std::shared_ptr<IR::Instruction>> m_statements;
	std::map<triton::usize, std::shared_ptr<IR::Expression>> m_expression_map; // associate symbolic variable with IR::Expression

	// known as symbolized registers
	std::map<triton::arch::Register, triton::engines::symbolic::SharedSymbolicVariable> known_regs;

	// probably there's much better way....
	std::map<std::shared_ptr<IR::Expression>, std::uint64_t> runtime_memory;

	//
	triton::uint64 next_handler_address, return_address;

	// <offset, <size, val>>
	std::map<triton::uint64, std::pair<triton::uint32, triton::uint64>> static_written;

	ThemidaHandlerContext()
	{
		this->address = 0;
		this->stack = 0;
		this->bytecode = 0;
		this->context = 0;
		this->exit_detected = 0;
		this->jmp_inside_detected = 0;
		this->jcc_detected = 0;
		this->execute_jcc = 1;
		this->next_handler_address = 0;
		this->return_address = 0;
	}
};
struct ThemidaCpuState
{
	// x86 cpustate
	triton::uint64 rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp;
	triton::uint64 r8, r9, r10, r11, r12, r13, r14, r15;

	// themida cpustate
	std::vector<triton::uint8> context_data;

	// vm information
	std::shared_ptr<ThemidaVMEnterContext> vmenter_ctx;
};


// ThemidaAnalyzer
class ThemidaAnalyzer
{
	struct binary_operation_pre
	{
		std::shared_ptr<IR::Expression> op0_expression, op1_expression;
	};

public:
	ThemidaAnalyzer(triton::arch::architecture_e arch = triton::arch::ARCH_X86);
	~ThemidaAnalyzer();

	// triton helpers
	bool is_x64() const;

	// returns rbp or ebp depends on architecture
	triton::arch::Register get_bp_register() const;

	// returns rsp or esp depends on architecture
	triton::arch::Register get_sp_register() const;

	// returns program counter
	triton::arch::Register get_ip_register() const;

	triton::uint64 get_bp() const;
	triton::uint64 get_sp() const;
	triton::uint64 get_ip() const;

	std::vector<triton::arch::Register> get_overlapped_regs(const triton::arch::Register& reg) const;

	// setConcreteMemoryAreaValue
	void load(AbstractStream& stream, triton::uint64 module_base, triton::uint64 virtual_address, triton::uint64 virtual_size);


	// vm-enter
	static void vm_enter_getConcreteMemoryValue_callback(triton::API& ctx, const triton::arch::MemoryAccess& mem);
	void symbolize_registers(std::shared_ptr<ThemidaVMEnterContext> context);
	void check_lock_xchg(triton::arch::Instruction& triton_instruction, std::shared_ptr<ThemidaVMEnterContext> context);
	void check_store_access(triton::arch::Instruction& triton_instruction, std::shared_ptr<ThemidaVMEnterContext> context);
	void check_eflags(triton::arch::Instruction& triton_instruction, std::shared_ptr<ThemidaVMEnterContext> context);
	void prepare_vm_enter(std::shared_ptr<ThemidaVMEnterContext> context);
	void analyze_vm_enter(AbstractStream& stream, std::shared_ptr<ThemidaVMEnterContext> ctx);


	// vm-handler
	std::shared_ptr<ThemidaCpuState> save_cpu_state(std::shared_ptr<ThemidaVMEnterContext> ctx);
	void load_cpu_state(const std::shared_ptr<ThemidaCpuState>& context, std::shared_ptr<ThemidaHandlerContext> ctx);


	bool symbolize_read_memory(const triton::arch::MemoryAccess& mem, std::shared_ptr<ThemidaHandlerContext> context);
	void storeAccess(triton::arch::Instruction& triton_instruction, std::shared_ptr<ThemidaHandlerContext> context);

	// save expressions before triton builds semantics
	std::vector<std::shared_ptr<IR::Expression>> save_expressions(triton::arch::Instruction& triton_instruction, std::shared_ptr<ThemidaHandlerContext> context);

	// x86 instruction -> IL instruction, symbolize eflags if needed
	//
	void check_arity_operation(triton::arch::Instruction& triton_instruction, const std::vector<std::shared_ptr<IR::Expression>>& operands_expressions, std::shared_ptr<ThemidaHandlerContext> context);

	void analyze_vm_handler_sub(AbstractStream& stream, triton::uint64 handler_address, std::shared_ptr<ThemidaHandlerContext> context);
	void analyze_vm_handler(AbstractStream& stream, triton::uint64 handler_address, std::shared_ptr<ThemidaHandlerContext> ctx);
	void analyze_vm_exit(std::shared_ptr<ThemidaHandlerContext> ctx);
	void analyze(AbstractStream& stream, triton::uint64 vmenter_address);


	void categorize_handler(std::shared_ptr<ThemidaHandlerContext> context);

	void print_output();

	std::shared_ptr<IR::Expression> get_vm_register(triton::uint64 offset, triton::uint32 size);

	// IR stuff
	void simplify_instructions(std::list<std::shared_ptr<IR::Instruction>>& instructions, bool basic_block = false);
	void simplify_statements(std::shared_ptr<ThemidaHandlerContext> context);

	// setConcreteMemoryAreaValue
	void setConcreteMemoryAreaValue(triton::uint64 address, const std::vector<unsigned char>& d);
	void setConcreteMemoryAreaValue(triton::uint64 address, const void* buf, unsigned int size);
	template <typename T>
	void setConcreteMemoryValue(triton::uint64 address, T data)
	{
		this->setConcreteMemoryAreaValue(address, &data, sizeof(T));
	}

private:
	// unique_ptr instead?
	std::shared_ptr<triton::API> triton_api;

	// for regs function
	mutable std::map<triton::arch::register_e, std::vector<triton::arch::Register>> m_overlapped_regs;

	// for VM Handler
	std::map<triton::uint64, std::shared_ptr<BasicBlock>> m_handlers;

	// <bytecode, statements>
	std::map<IR::Label, std::list<std::shared_ptr<IR::Instruction>>> m_statements;

	std::map<triton::uint64, std::shared_ptr<IR::Expression>> m_vm_regs;

	// <bytecode, <handler_address, cpu_state>>
	std::map<IR::Label, std::pair<triton::uint64, std::shared_ptr<ThemidaCpuState>>> m_destinations;
};