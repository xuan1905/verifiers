SIMPLE_PROMPT = """
Respond in the following format, using careful step-by-step reasoning.

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

CODE_PROMPT = """\
Given a math problem, use step-by-step reasoning and code execution to solve the problem. 

In each step, first think step-by-step about how to solve the problem inside <reasoning> tags. \
Then, write Python code inside <code> tags to work out calculations. \
The <code> tag should only contain code that can be executed by a Python interpreter. \
You may import numpy, scipy, and sympy libraries for your calculations. \
You will then be shown the output of print statements from your code in <output> tags.\
Variables do not persist across <code> calls and should be redefined each time.

Continue this process until you are confident that you have found the solution. \
Then, summarize your reasoning in <reasoning> tags, and give your final answer in <answer> tags. """