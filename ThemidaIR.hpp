#pragma once

namespace IR
{
	// declare
	class Expression;
	typedef std::shared_ptr<Expression> exprptr;


	enum expression_type
	{
		expr_register,
		expr_memory,
		expr_variable,
		expr_immediate,
		expr_unary_operation,
		expr_binary_operation
	};

	enum class unary_op
	{
		invalid = -1,
		inc,
		dec,
		not_,
		neg,
		extend,
		signextend,
		zeroextend,
		extract,
		flagsof,
	};

	enum class binary_op
	{
		invalid = -1,
		add,
		sub,
		imul,
		shl,
		shr,
		rcr,
		rcl,
		rol,
		ror,
		and_,
		or_,
		xor_,
		cmp,
		test
	};

	// Expression
	class Immediate;
	class Expression
	{
	protected:
		expression_type m_type;

	protected:
		Expression(expression_type type) : m_type(type)
		{
		}
		virtual ~Expression() {}

	public:
		expression_type get_type() const
		{
			return this->m_type;
		}

	public:
		virtual triton::uint32 get_bit_size() const = 0;
		virtual triton::uint32 get_size() const = 0;
		virtual void to_string(std::ostream& stream) const = 0;

		virtual triton::uint64 get_value() const
		{
			throw std::runtime_error(__FUNCTION__);
		}
		virtual exprptr get_operand(unsigned int i) const
		{
			throw std::runtime_error(__FUNCTION__);
		}
		virtual void set_operand(unsigned int i, exprptr expr)
		{
			throw std::runtime_error(__FUNCTION__);
		}
	};
	std::ostream& operator<<(std::ostream& stream, const Expression& expr);
	std::ostream& operator<<(std::ostream& stream, const Expression* expr);

	extern std::shared_ptr<Expression> simplify_expression(const std::shared_ptr<Expression>& expression);


	// Register
	class Register : public Expression
	{
	public:
		// x86register
		Register(const triton::arch::Register& triton_register);

		virtual triton::uint32 get_bit_size() const override;
		virtual triton::uint32 get_size() const override;
		virtual void to_string(std::ostream& stream) const override;

		std::string get_name() const;
		triton::uint64 get_offset() const;
		void set_offset(triton::uint64 offset);

	private:
		std::string m_name;
		triton::uint64 m_offset;
		triton::arch::Register m_register;
	};


	// Memory
	class Memory : public Expression
	{
	public:
		Memory(const std::shared_ptr<Expression>& expr, const triton::arch::Register& segment_register, triton::uint32 size);

		virtual triton::uint32 get_bit_size() const override;
		virtual triton::uint32 get_size() const override;
		virtual void to_string(std::ostream& stream) const override;

		std::shared_ptr<Expression> get_expression() const;
		void set_expression(std::shared_ptr<Expression> expr);

	private:
		triton::uint32 m_size;
		triton::arch::Register m_segment_register;
		std::shared_ptr<Expression> m_expr;
	};


	// Variable
	class Variable : public Expression
	{
	public:
		Variable(triton::uint32 size);
		Variable(const std::string& name, triton::uint32 size);

		virtual triton::uint32 get_bit_size() const override;
		virtual triton::uint32 get_size() const override;
		virtual void to_string(std::ostream& stream) const override;

		std::string get_name() const;

		static std::shared_ptr<Variable> create_variable(triton::uint32 size);

	private:
		static int s_index;
		std::string m_name;
		triton::uint32 m_size;
	};


	// Immediate
	class Immediate : public Expression
	{
	public:
		Immediate(triton::uint64 value, triton::uint32 size = 4);

		virtual triton::uint32 get_bit_size() const override;
		virtual triton::uint32 get_size() const override;
		virtual void to_string(std::ostream& stream) const override;

		triton::uint64 get_value() const override
		{
			return this->m_immediate;
		}

	private:
		triton::uint32 m_size;
		triton::uint64 m_immediate;
	};



	// UnaryOperation
	class UnaryOperation : public Expression
	{
	protected:
		UnaryOperation(const std::shared_ptr<Expression>& op, unary_op t);
		exprptr m_op;
		unary_op m_type;

	public:
		virtual triton::uint32 get_bit_size() const override;
		virtual triton::uint32 get_size() const override;
		exprptr get_operand(unsigned int i) const override
		{
			if (i != 0)
				throw std::runtime_error(__FUNCTION__);
			return this->m_op;
		}
		void set_operand(unsigned int i, exprptr expr) override
		{
			if (i != 0)
				throw std::runtime_error(__FUNCTION__);
			this->m_op = expr;
		}
	};


	// BinaryOperation
	class BinaryOperation : public Expression
	{
	protected:
		BinaryOperation(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1, binary_op t);

		std::shared_ptr<Expression> m_op0, m_op1;
		binary_op m_binary_type;

	public:
		virtual triton::uint32 get_bit_size() const override;
		virtual triton::uint32 get_size() const override;
		virtual void to_string(std::ostream& stream) const = 0;
		exprptr get_operand(unsigned int i) const override
		{
			if (i == 0) return this->m_op0;
			else if (i == 1) return this->m_op1;
			throw std::runtime_error(__FUNCTION__);
		}
		void set_operand(unsigned int i, exprptr expr) override
		{
			if (i == 0)
				this->m_op0 = expr;
			else if (i == 1)
				this->m_op1 = expr;
			else
				throw std::runtime_error(__FUNCTION__);
		}

		binary_op get_binary_type() const;
	};


	// unary
	class Inc : public UnaryOperation
	{
	public:
		Inc(const std::shared_ptr<Expression>& op) : UnaryOperation(op, unary_op::inc)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Inc(" << this->m_op << ")";
		}
	};
	class Dec : public UnaryOperation
	{
	public:
		Dec(const std::shared_ptr<Expression>& op) : UnaryOperation(op, unary_op::dec)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Dec(" << this->m_op << ")";
		}
	};
	class Not : public UnaryOperation
	{
	public:
		Not(const std::shared_ptr<Expression>& op) : UnaryOperation(op, unary_op::not_)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Not(" << this->m_op << ")";
		}
	};
	class Neg : public UnaryOperation
	{
	public:
		Neg(const std::shared_ptr<Expression>& op) : UnaryOperation(op, unary_op::neg)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Neg(" << this->m_op << ")";
		}
	};

	// binary
	class Add : public BinaryOperation
	{
	public:
		Add(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1) : BinaryOperation(op0, op1, binary_op::add)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Add(" << this->m_op0 << ", " << this->m_op1 << ")";
		}
	};
	class Sub : public BinaryOperation
	{
	public:
		Sub(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1) : BinaryOperation(op0, op1, binary_op::sub)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Sub(" << this->m_op0 << ", " << this->m_op1 << ")";
		}
	};

	class Imul : public BinaryOperation
	{
	public:
		Imul(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1) : BinaryOperation(op0, op1, binary_op::imul)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Imul(" << this->m_op0 << ", " << this->m_op1 << ")";
		}
	};

	class Shl : public BinaryOperation
	{
	public:
		Shl(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1) : BinaryOperation(op0, op1, binary_op::shl)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Shl(" << this->m_op0 << ", " << this->m_op1 << ")";
		}
	};
	class Shr : public BinaryOperation
	{
	public:
		Shr(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1) : BinaryOperation(op0, op1, binary_op::shr)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Shr(" << this->m_op0 << ", " << this->m_op1 << ")";
		}
	};

	class Rcr : public BinaryOperation
	{
	public:
		Rcr(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1) : BinaryOperation(op0, op1, binary_op::rcr)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Rcr(" << this->m_op0 << ", " << this->m_op1 << ")";
		}
	};
	class Rcl : public BinaryOperation
	{
	public:
		Rcl(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1) : BinaryOperation(op0, op1, binary_op::rcl)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Rcl(" << this->m_op0 << ", " << this->m_op1 << ")";
		}
	};
	class Rol : public BinaryOperation
	{
	public:
		Rol(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1) : BinaryOperation(op0, op1, binary_op::rol)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Rol(" << this->m_op0 << ", " << this->m_op1 << ")";
		}
	};
	class Ror : public BinaryOperation
	{
	public:
		Ror(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1) : BinaryOperation(op0, op1, binary_op::ror)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Ror(" << this->m_op0 << ", " << this->m_op1 << ")";
		}
	};

	class And : public BinaryOperation
	{
	public:
		And(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1) : BinaryOperation(op0, op1, binary_op::and_)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "And(" << this->m_op0 << ", " << this->m_op1 << ")";
		}
	};
	class Or : public BinaryOperation
	{
	public:
		Or(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1) : BinaryOperation(op0, op1, binary_op::or_)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Or(" << this->m_op0 << ", " << this->m_op1 << ")";
		}
	};
	class Xor : public BinaryOperation
	{
	public:
		Xor(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1) : BinaryOperation(op0, op1, binary_op::xor_)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Xor(" << this->m_op0 << ", " << this->m_op1 << ")";
		}
	};

	class Cmp : public BinaryOperation
	{
	public:
		Cmp(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1) : BinaryOperation(op0, op1, binary_op::cmp)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Cmp(" << this->m_op0 << ", " << this->m_op1 << ")";
		}
	};
	class Test : public BinaryOperation
	{
	public:
		Test(const std::shared_ptr<Expression>& op0,
			const std::shared_ptr<Expression>& op1) : BinaryOperation(op0, op1, binary_op::test)
		{

		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Test(" << this->m_op0 << ", " << this->m_op1 << ")";
		}
	};

	// Pseudo
	class Extend : public UnaryOperation
	{
	public:
		Extend(const std::shared_ptr<Expression>& op0, triton::uint32 size) : UnaryOperation(op0, unary_op::extend)
		{
			this->m_size = size;
		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Extend(" << this->m_op << ")";
		}

	private:
		triton::uint32 m_size;
	};
	class SignExtend : public UnaryOperation
	{
	public:
		SignExtend(const std::shared_ptr<Expression>& op0) : UnaryOperation(op0, unary_op::signextend)
		{
		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "SignExtend(" << this->m_op << ")";
		}
	};
	class ZeroExtend : public UnaryOperation
	{
	public:
		ZeroExtend(const std::shared_ptr<Expression>& op0) : UnaryOperation(op0, unary_op::zeroextend)
		{
		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "ZeroExtend(" << this->m_op << ")";
		}
	};

	// Extract(eax, 7, 0) -> al, Extract(eax, 15, 8) -> ah, Extract(eax, 15, 0) -> ax
	class Extract : public UnaryOperation
	{
	public:
		Extract(const std::shared_ptr<Expression>& op0, triton::uint32 high, triton::uint32 low = 0) 
			: UnaryOperation(op0, unary_op::extract)
		{
			this->m_high = high;
			this->m_low = low;
		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "Extract(" << this->m_op << "," << this->m_high << "," << this->m_low << ")";
		}

	private:
		triton::uint32 m_high, m_low;
	};

	// 
	class Flags : public UnaryOperation
	{
	public:
		Flags(const std::shared_ptr<Expression>& op0) : UnaryOperation(op0, unary_op::flagsof)
		{
		}

		virtual void to_string(std::ostream& stream) const override
		{
			stream << "FlagsOf(" << this->m_op << ")";
		}

	private:
	};
}