#pragma once

#include "Label.hpp"

namespace IR
{
	struct handler_instructions
	{
		Label vip;
		std::list<std::shared_ptr<IR::Instruction>> instructions;
	};

	struct BB
	{
		Label label;
		
		//std::list<std::shared_ptr<IR::Instruction>> instructions;
		std::list<handler_instructions> handler_objects;


		bool terminator;
		std::shared_ptr<BB> next_basic_block, target_basic_block;

	public:
		std::string to_string() const
		{
			return this->label.to_string();
		}
	};
}