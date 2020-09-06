#pragma once

#include "AbstractStream.hpp"
#include "Label.hpp"


#define THEMIDA_CONTEXT_SIZE	0x200


// forward declaration
struct BasicBlock;
namespace IR
{
	class Expression;
	class Register;
	class Instruction;
}

// ctx
struct vm_objects
{
	triton::uint64 context_addr, lock_addr, bytecode_addr;
};
struct vmenter_objects
{
	// base address / lock / pcode
	triton::uint64 context_addr, lock_addr, bytecode_addr, bytecode;

	// <runtime_address, <value, size>>
	std::map<triton::uint64, std::pair<triton::uint64, triton::uint32>> initial_data;

	// handler address that will be executed after vmenter
	triton::uint64 first_handler_address;

	// instructions lifted
	std::list<std::shared_ptr<IR::Instruction>> instructions;

	// internal usages
	struct
	{
		triton::uint64 original_sp, modified_sp;
		bool eflags_written;
		// <triton_symid, expression>
		std::map<triton::usize, std::shared_ptr<IR::Register>> registers;
	};

	vmenter_objects() {}
	~vmenter_objects() {}
};
struct ThemidaHandlerContext
{
	// information gathered from vm enter
	triton::uint64 context_addr, lock_addr, bytecode_addr;

	// before start
	triton::uint64 address;
	triton::uint64 stack, bytecode, context;
	triton::engines::symbolic::SharedSymbolicVariable symvar_stack, symvar_bytecode, symvar_context;
	bool execute_jcc;

	// <triton_symid, symvar>
	std::map<triton::usize, triton::engines::symbolic::SharedSymbolicVariable> vmregs;

	// ret
	bool exit_detected, jmp_inside_detected, jcc_detected;

	// expressions
	std::list<std::shared_ptr<IR::Instruction>> instructions;
	std::map<triton::usize, std::shared_ptr<IR::Expression>> m_expression_map; // associate symbolic variable with IR::Expression

	// known as symbolized registers
	std::map<triton::arch::Register, triton::engines::symbolic::SharedSymbolicVariable> known_regs;

	// probably there's much better way....
	std::map<std::shared_ptr<IR::Expression>, std::uint64_t> runtime_memory;

	//
	triton::uint64 next_handler_address, next_bytecode, return_address;

	// <offset, <size, val>>
	std::map<triton::uint64, std::pair<triton::uint32, triton::uint64>> static_written;

public:
	ThemidaHandlerContext(triton::uint64 context_addr, triton::uint64 lock_addr, triton::uint64 bytecode_addr)
	{
		this->context_addr = context_addr;
		this->lock_addr = lock_addr;
		this->bytecode_addr = bytecode_addr;

		this->address = 0;
		this->stack = 0;
		this->bytecode = 0;
		this->context = 0;
		this->exit_detected = 0;
		this->jmp_inside_detected = 0;
		this->jcc_detected = 0;
		this->execute_jcc = 1;
		this->next_handler_address = 0;
		this->next_bytecode = 0;
		this->return_address = 0;
	}

	void reset()
	{
		this->address = 0;
		this->stack = 0;
		this->bytecode = 0;
		this->context = 0;
		this->symvar_stack.reset();
		this->symvar_bytecode.reset();
		this->symvar_context.reset();
		this->execute_jcc = true;

		this->vmregs.clear();
		this->instructions.clear();
		this->m_expression_map.clear();
		this->known_regs.clear();
		this->runtime_memory.clear();
		this->static_written.clear();

		this->exit_detected = false;
		this->jmp_inside_detected = false;
		this->jcc_detected = false;
		this->next_handler_address = 0;
		this->next_bytecode = 0;
		this->return_address = 0;
	}

	std::shared_ptr<IR::Expression> triton_to_expr(triton::engines::symbolic::SharedSymbolicVariable symvar) const
	{
		if (!symvar)
			throw std::runtime_error("nope!");

		auto it = this->m_expression_map.find(symvar->getId());
		if (it == this->m_expression_map.end())
			throw std::runtime_error("nope!");

		return it->second;
	}
	std::shared_ptr<IR::Expression> triton_to_expr(triton::usize symvar_id) const
	{
		auto it = this->m_expression_map.find(symvar_id);
		if (it == this->m_expression_map.end())
			throw std::runtime_error("nope!");
		return it->second;
	}
};
struct ThemidaCpuState
{
	// x86 cpustate
	triton::uint64 rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp;
	triton::uint64 r8, r9, r10, r11, r12, r13, r14, r15;

	// themida cpustate
	std::vector<triton::uint8> context_data;

	// not really cpu state but i need it
	triton::uint64 context_addr, lock_addr, bytecode_addr;
};



typedef std::shared_ptr<ThemidaHandlerContext> hdlr_ctx_ptr;


// ThemidaAnalyzer
class ThemidaAnalyzer
{
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

	// setConcreteMemoryAreaValue
	void setConcreteMemoryAreaValue(triton::uint64 address, const std::vector<unsigned char>& values);
	void setConcreteMemoryAreaValue(triton::uint64 address, const void* buf, unsigned int size);

	template <typename T>
	void setConcreteMemoryValue(triton::uint64 address, const T data)
	{
		static_assert(std::is_arithmetic<T>::value, "");
		this->setConcreteMemoryAreaValue(address, (const triton::uint8*)&data, sizeof(T));
	}

	// returns overlapped registers (include input)
	std::vector<triton::arch::Register> get_overlapped_regs(const triton::arch::Register& reg) const;


	// functions for vm enter
	void vmenter_symbolize_registers(vmenter_objects& ctx);
	void vmenter_check_lock_xchg(triton::arch::Instruction& triton_instruction, vmenter_objects& ctx);
	void vmenter_check_store_access(triton::arch::Instruction& triton_instruction, vmenter_objects& ctx);
	void vmenter_check_eflags(triton::arch::Instruction& triton_instruction, vmenter_objects& ctx);
	void vmenter_prepare(vmenter_objects& ctx);
	void lift_vm_enter(AbstractStream& stream, triton::uint64 address, vmenter_objects& ctx);

	// save/load vm cpu state
	std::shared_ptr<ThemidaCpuState> save_cpu_state();
	void load_cpu_state(const std::shared_ptr<ThemidaCpuState>& context);


	// functions for vm handler
	bool symbolize_read_memory(const triton::arch::MemoryAccess& mem, std::shared_ptr<ThemidaHandlerContext> context);
	void storeAccess(triton::arch::Instruction& triton_instruction, std::shared_ptr<ThemidaHandlerContext> context);

	// save expressions before triton builds semantics
	std::vector<std::shared_ptr<IR::Expression>> save_expressions(triton::arch::Instruction& triton_instruction, std::shared_ptr<ThemidaHandlerContext> context);

	// x86 instruction -> IL instruction, symbolize eflags if needed
	void check_arity_operation(triton::arch::Instruction& triton_instruction, const std::vector<std::shared_ptr<IR::Expression>>& operands_expressions, hdlr_ctx_ptr ctx);


	void run_vm_handler(AbstractStream& stream, triton::uint64 handler_address, hdlr_ctx_ptr ctx);
	std::list<std::shared_ptr<IR::Instruction>> lift_vm_exit(hdlr_ctx_ptr ctx);
	std::list<std::shared_ptr<IR::Instruction>> vmhandler_post(hdlr_ctx_ptr ctx);

	void vmhandler_prepare(triton::uint64 handler_address, hdlr_ctx_ptr ctx);
	void lift_vm_handler(AbstractStream& stream, triton::uint64 handler_address, hdlr_ctx_ptr ctx);


	// main
	typedef std::map<IR::Label, std::pair<triton::uint64, std::shared_ptr<ThemidaCpuState>>> vm_label_t;
	void explore_handler(AbstractStream& stream, triton::uint64 handler_address, std::set<IR::Label>& visit, vm_label_t& labels, hdlr_ctx_ptr handler_ctx);
	void explore_entry(AbstractStream& stream, triton::uint64 vmenter_address, std::set<IR::Label>& visit, vm_label_t& labels);
	void analyze(AbstractStream& stream, triton::uint64 vmenter_address);
	void print_output();

	std::shared_ptr<IR::Expression> get_vm_register(triton::uint64 offset, triton::uint32 size);


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
};