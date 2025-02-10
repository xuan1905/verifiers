from verifiers.parsers import XMLParser

math_parser = XMLParser(fields=["reasoning", "answer"])
MATH_FEW_SHOT = [
    [
        {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
        {'role': 'assistant', 'content': math_parser.format(
            reasoning='The largest single-digit prime number is 7.',
            answer='7'
        )}
    ]
]

DOUBLECHECK_FEW_SHOT = [
    [
        {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
        {'role': 'assistant', 'content': math_parser.format(
            reasoning='The largest single-digit prime number is 7.',
            answer='7'
        )},
        {'role': 'user', 'content': 'Are you sure?'},
        {'role': 'assistant', 'content': math_parser.format(
            reasoning='The only larger single-digit numbers are 8 and 9, which are not prime. So yes, the answer is 7.',
            answer='7'
        )}
    ]
]

code_parser = XMLParser(fields=["reasoning", ("code", "answer")])
output_parser = XMLParser(fields=["output"])
MATH_CODE_FEW_SHOT = [
    [
        {'role': 'user', 'content': 'What is sum of the first 100 positive even integers?'},
        {'role': 'assistant', 'content': code_parser.format(
            reasoning='Let\'s compute the sum of the first 100 positive even integers.',
            code='print(sum(range(2, 102, 2)))'
        )},
        {'role': 'user', 'content': output_parser.format(output='2550')},
        {'role': 'assistant', 'content': code_parser.format(
            reasoning='The answer is 2550.',
            answer='2550'
        )}
    ]
]