#pragma once

extern triton::engines::symbolic::SharedSymbolicVariable get_symbolic_var(const triton::ast::SharedAbstractNode &node);
extern std::set<triton::ast::SharedAbstractNode> collect_symvars(const triton::ast::SharedAbstractNode &parent);

extern bool is_unary_operation(const triton::arch::Instruction &triton_instruction);
extern bool is_binary_operation(const triton::arch::Instruction &triton_instruction);

// return true if inst is (mov | movsx | movzx)
extern bool is_mov_operation(const triton::arch::Instruction& triton_instruction);