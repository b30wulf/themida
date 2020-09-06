#pragma once

namespace IR
{
	class Instruction;
	struct BB;
}

extern void simplify_xchg(std::shared_ptr<IR::BB> bb);

extern void simplify_instructions(std::list<std::shared_ptr<IR::Instruction>>& instructions, bool basic_block = false);