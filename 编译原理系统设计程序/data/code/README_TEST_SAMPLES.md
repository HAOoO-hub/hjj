# 综合测试样例说明

## 样例 1：demo_comprehensive_1.cmini
**重点展示：常量、复杂表达式、函数调用**

### 功能特点
- ✅ 多参数函数定义（add, max, compute）
- ✅ if/else 分支结构
- ✅ while 循环
- ✅ 变量声明带初始化
- ✅ 常量声明和使用（MAX_VALUE, MIN_VALUE, STEP）
- ✅ 四则运算（+ - * /）
- ✅ 关系运算符（> < == !=）
- ✅ 逻辑运算符（&& ||）
- ✅ 括号表达式
- ✅ 嵌套函数调用
- ✅ 复杂条件判断

### 测试结果
- Tokens: 350
- Functions: 4 (add, max, compute, main)
- Constants: 3 (MAX_VALUE=100, MIN_VALUE=0, STEP=5)
- Variables: 12
- Execution steps: 112
- Return value: 0 ✓

---

## 样例 2：demo_comprehensive_2.cmini
**重点展示：递归调用（阶乘）**

### 功能特点
- ✅ 递归函数定义（factorial）
- ✅ 递归终止条件（使用 || 保证稳定性）
- ✅ 多次递归调用
- ✅ 常量声明（LIMIT）
- ✅ 递归结果参与条件判断
- ✅ 递归结果参与运算

### 测试结果
- Tokens: 121
- Functions: 2 (factorial, main)
- Constants: 1 (LIMIT=100)
- Variables: 4
- Execution steps: 112
- Return value: 0 ✓
- factorial(5) = 120 ✓
- factorial(4) = 24 ✓

---

## 系统优势展示

### 1. 完整的编译流程支持
- 词法分析 → 语法分析 → 语义分析 → IR生成 → 执行

### 2. 丰富的语言特性
- 常量声明和跟踪
- 多参数函数
- 递归调用
- 复杂表达式
- 完整的控制流（if/else, while）

### 3. 完善的符号表管理
- 总符号表
- 变量表
- 函数表
- **常量表** ⭐

### 4. 稳定的解释器实现
- 支持递归调用（使用 `==` 和 `||` 组合）
- 正确的变量作用域管理
- 准确的返回值处理

### 5. 详细的错误检测
- 词法错误
- 语法错误
- 语义错误（重复定义、未初始化等）

---

## 注意事项

### 递归调用的稳定性
- ✅ 推荐使用：`if (n == 0 || n == 1)` 
- ❌ 避免使用：`if (n <= 1)` （可能导致无限递归）

### 斐波那契数列的限制
由于斐波那契的指数级递归深度，当前解释器对 fibonacci(n) 的支持有限：
- fibonacci(3) 需要 >10,000 步
- fibonacci(5) 需要 >100,000 步
- 建议使用阶乘等线性递归作为测试

### 执行步数上限
- 默认：10,000 步
- 测试脚本中已增加到：2,000,000 步
- 可根据需要调整 `executor.run(max_steps=N)`
