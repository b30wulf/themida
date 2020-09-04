#pragma once

namespace IR
{
	enum class e_label
	{
		invalid,
		x86_ip,
		vm_ip,
	};

	// vmenter / vminternal / vmexit
	class Label
	{
		e_label m_type;
		triton::uint64 m_addr;

	public:
		Label()
		{
			this->m_addr = 0;
			this->m_type = e_label::invalid;
		}
		Label(triton::uint64 address, e_label e)
		{
			this->m_addr = address;
			this->m_type = e;
		}
		~Label() {}


		// operators
		bool operator==(const Label& label) const
		{
			return this->m_type == label.m_type
				&& this->m_addr == label.m_addr;
		}
		bool operator!=(const Label& label) const
		{
			return this->m_type != label.m_type
				|| this->m_addr != label.m_addr;
		}
		bool operator<(const Label& label) const
		{
			if (this->m_type == label.m_type)
				return this->m_addr < label.m_addr;
			else
				return this->m_type < label.m_type;
		}

		//
		bool is_valid() const
		{
			return this->m_type != e_label::invalid;
		}
		bool is_vip() const
		{
			return this->m_type == e_label::vm_ip;
		}


		//
		std::string to_string() const
		{
			char buf[32];
			if (this->m_type == e_label::x86_ip)
			{
				// virtual address
				sprintf_s(buf, 32, "%llx", this->m_addr);
			}
			else
			{
				// vm
				sprintf_s(buf, 32, "b-%llx", this->m_addr);
			}
			return buf;
		}

		// constructors
		static Label x86(triton::uint64 addr)
		{
			return Label(addr, e_label::x86_ip);
		}
		static Label vip(triton::uint64 addr)
		{
			return Label(addr, e_label::vm_ip);
		}
	};
}