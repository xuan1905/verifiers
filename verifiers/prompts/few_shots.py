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
CODE_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'What is sum of the first 100 positive even integers?'
        },
        {
            'role': 'assistant',
            'content': code_parser.format(
                reasoning='Let\'s compute the sum of the first 100 positive even integers.',
                code='print(sum(range(2, 102, 2)))'
            )
        },
        {
            'role': 'user', 
            'content': output_parser.format(output='2550')
        },
        {
            'role': 'assistant',
            'content': code_parser.format(reasoning='The answer is 2550.', answer='2550')
        },
        {
            'role': 'user',
            'content': 'What is the sum of the first 100 natural numbers, minus the largest prime number less than 100?'
        },
        {
            'role': 'assistant',
            'content': code_parser.format(
                reasoning='The sum of the first n natural numbers is given by the formula n(n+1)/2.',
                code='print(100*101/2)'
            )
        },
        {
            'role': 'user',
            'content': output_parser.format(output='5050')
        },
        {
            'role': 'assistant',
            'content': code_parser.format(
                reasoning='The sum of the first 100 natural numbers is 5050. Now we need to subtract the largest prime number less than 100.',
                code='print(5050 - 97)'
            )
        },
        {
            'role': 'user',
            'content': output_parser.format(output='4953')
        },
        {
            'role': 'assistant',
            'content': code_parser.format(
                reasoning='The largest prime number less than 100 is 97. Subtracting this from 5050 gives 4953.',
                answer='4953'
            )
        }
    ]
]

tool_parser = XMLParser(fields=["reasoning", ("tool", "answer")])
result_parser = XMLParser(fields=["result"])

TOOL_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'What is the current working directory?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Let\'s use the pwd command to find out the current working directory.',
                tool='pwd'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result='/Users/user/project')
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='The current working directory is /Users/user/project.',
                answer='/Users/user/project'
            )
        },
        {
            'role': 'user',
            'content': 'How many Python files are in the current directory and its subdirectories?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Let\'s use the find command to count Python files.',
                tool='find . -name "*.py" | wc -l'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result='42')
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='There are 42 Python files in the current directory and its subdirectories.',
                answer='42'
            )
        }
    ]
]

COMMONSENSE_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'Which would be louder: a mouse or an elephant?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Let\'s compare the volume levels of a mouse and an elephant.',
                tool='compare mouse elephant volume'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result='''{
  "difference": -4,
  "mouse_volume": 1,
  "elephant_volume": 5
}''')
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Based on the comparison, an elephant has a volume level of 5 while a mouse has a volume level of 1 (on a scale of 1-5). The difference of -4 indicates the elephant is much louder.',
                answer='An elephant would be louder than a mouse.'
            )
        },
        {
            'role': 'user',
            'content': 'What properties does a car have?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Let\'s look up the properties of a car.',
                tool='get_related car'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result='''{
  "properties": ["metallic", "fast", "loud", "heavy"],
  "category": "vehicle"
}''')
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='A car has several key properties: it is metallic, fast, loud, and heavy. It belongs to the category of vehicles.',
                answer='A car is a vehicle that is metallic, fast, loud, and heavy.'
            )
        }
    ]
]