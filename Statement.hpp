#pragma once

#include "Label.hpp"

class x86_instruction;

namespace IR
{
	class Expression;
	typedef std::shared_ptr<Expression> exprptr;

	enum class instruction_id
	{
		invalid = -1,
		assign,
		push,
		pop,
		ret,
		jcc,
		jmp,
		xchg,

		// x86 asm
		xed,
		
		// for dbg
		lb
	};

	class Instruction
	{
	protected:
		instruction_id m_instruction_id;
		std::vector<exprptr> m_expressions;

		Instruction(instruction_id id);
		virtual ~Instruction();

	public:
		instruction_id get_id() const;

	public:
		virtual void to_string(std::ostream& stream) const = 0;

		virtual exprptr get_lhs() const
		{
			if (this->m_expressions.empty())
				throw std::runtime_error("instruction has no left hand side");
			return this->m_expressions[0];
		}
		virtual exprptr get_rhs() const
		{
			if (this->m_expressions.size() != 2)
				throw std::runtime_error("instruction has no right hand side");
			return this->m_expressions[1];
		}
		virtual void set_lhs(const exprptr& lhs)
		{
			if (this->m_expressions.empty())
				throw std::runtime_error("instruction has no left hand side");
			this->m_expressions[0] = lhs;
		}
		virtual void set_rhs(const exprptr& rhs)
		{
			if (this->m_expressions.size() != 2)
				throw std::runtime_error("instruction has no right hand side");
			this->m_expressions[1] = rhs;
		}

		virtual exprptr get_expression() const
		{
			if (this->m_expressions.empty())
				throw std::runtime_error("instruction has no expression");
			return this->m_expressions[0];
		}
		virtual void set_expression(const exprptr& expr)
		{
			if (this->m_expressions.empty())
				throw std::runtime_error("instruction has no expression");
			this->m_expressions[0] = expr;
		}
	};

	// Assign (x = y)
	class Assign : public Instruction
	{
	public:
		Assign(const exprptr& lhs, const exprptr& rhs);

		virtual void to_string(std::ostream& stream) const override;
	};

	// Push (push x)
	class Push : public Instruction
	{
	public:
		Push(const exprptr& expr);

		void to_string(std::ostream& stream) const override;
	};

	// Pop (pop x)
	class Pop : public Instruction
	{
	public:
		Pop(const exprptr& expr);

		void to_string(std::ostream& stream) const override;
	};


	// Xchg (xchg x, y) tiger specific?
	class Xchg : public Instruction
	{
	public:
		Xchg(const exprptr& lhs, const exprptr& rhs);

		void to_string(std::ostream& stream) const override;
	};

	class LB : public Instruction
	{
		Label m_label;

	public:
		LB(const Label& label) : Instruction(instruction_id::lb) { m_label = label; }

		void to_string(std::ostream& stream) const override
		{
			stream << m_label.to_string() << ":";
		}
	};

	//
	class Jcc : public Instruction
	{
	public:
		Jcc(const Label& label) : Instruction(instruction_id::jcc)
		{
			this->m_label = label;
		}

		Label get_destination() const
		{
			return this->m_label;
		}

		void to_string(std::ostream& stream) const override
		{
			stream << "jcc " << this->m_label.to_string();
		}

	private:
		Label m_label;
	};

	class Jmp : public Instruction
	{
	public:
		Jmp(const Label& label) : Instruction(instruction_id::jmp)
		{
			this->m_label = label;
		}

		Label get_destination() const
		{
			return this->m_label;
		}

		void to_string(std::ostream& stream) const override
		{
			stream << "jmp " << this->m_label.to_string();
		}

	private:
		Label m_label;
	};

	class Ret : public Instruction
	{
	public:
		Ret(int offset = 0) : Instruction(instruction_id::ret)
		{
			this->m_offset = offset;
		}

		void to_string(std::ostream& stream) const override
		{
			stream << "ret " << std::hex << this->m_offset << std::dec;
		}

	private:
		int m_offset;
	};

	class X86Instruction : public Instruction
	{
	public:
		X86Instruction(std::shared_ptr<x86_instruction> xed_instruction) : Instruction(instruction_id::xed)
		{
			this->m_instruction = xed_instruction;
		}

		void to_string(std::ostream& stream) const override
		{
			stream << this->m_instruction;
		}

	private:
		std::shared_ptr<x86_instruction> m_instruction;
	};

	std::ostream& operator<<(std::ostream& stream, const Instruction& statement);
	std::ostream& operator<<(std::ostream& stream, const Instruction* statement);
}