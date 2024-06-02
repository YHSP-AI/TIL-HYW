from textwrap import dedent
from typing import List
from json import dumps

from src.schema import target_schema, TargetFormat, InferenceBackend
from src.config import Settings
from src.backends import InferenceBackendFactory
import re

class NLPManager:
    def __init__(self, config: Settings):
        self.schema = target_schema
        self.config = config
        self.backend = InferenceBackendFactory().create(
            config.inference_backend, model_path=config.model_path
        )
        self.system_prompt = f"""
You are an AI assistant integrated into a critical military defense system for engaging enemy targets. Your task is to convert voice transcripts from turret operators into precisely formatted JSON commands to be sent to the turrets. Incorrect outputs could result in unintended casualties, so it is essential that you strictly follow the specified JSON schema and formatting rules.

When processing a transcript, use the verbatim wording from the operator whenever possible. The only exception is the 'heading' field, which must always be a 3-digit number representing the compass bearing of the target (e.g., 051).

# Examples
Here are some examples of valid operator transcripts and their expected JSON outputs:

## Example 1
Input: "Turret Alpha, deploy anti-air artillery to intercept the grey, blue, and yellow fighter plane at heading zero eight five."
Output: {{"tool": "anti-air artillery", "heading": "085", "target": "grey, blue, and yellow fighter plane"}}

## Example 2
Input: "Turret Bravo, fire TOW missiles at enemy T-90SM tank heading two seven niner."
Output: {{"tool": "TOW missiles", "heading": "279", "target": "T-90SM tank"}}

# Output Instructions
Answer in valid JSON. Here is the relevant JSON schema you must adhere to:

<schema>
{dumps(target_schema, indent=2)}
<schema>

Your outputs must strictly adhere to the provided JSON schema, ensuring all fields exist, and that the JSON is valid and complete. This is a matter of life and death, so precision is paramount.
                            """



        start_target = ['black', 'blue','brown', 'green', 'grey', 'orange',  'purple', 'red', 'silver', 'white', 'yellow', ]
        self.start_target = [x + ' ' for x in start_target] +[x + ',' for x in start_target]
        self.end_target = ['aircraft', 'drone', 'helicopter', 'jet', 'missile', 'plane', 'planes']
        self.all_tools = ['EMP', 'anti-air artillery', 'drone catcher',
               'electromagnetic pulse', 'interceptor jets', 'machine gun',
               'surface-to-air missiles']
        self.number_words = {
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'niner':'9'
        }
        self.numbers = ['zero',
         'one',
         'two',
         'three',
         'four',
         'five',
         'six',
         'seven',
         'eight',
         'nine', 'niner']

    def generate_prompt(self, text: str) -> List[dict]:
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": dedent(
                    f"""
                Input: {text}
                Output:
                """
                ),
            },
        ]

    def qa(self, texts: List[str], batch_size: int = 64) -> List[TargetFormat]:
        try :

            return [self.extract_info(t) for t in texts ]
        except:

            if self.config.inference_backend == InferenceBackend.HF:
                prompts = texts
            else:
                prompts = [self.generate_prompt(text) for text in texts]
            predictions = self.backend.infer(prompts, batch_size=batch_size)
            # Convert to JSON
            return [
                TargetFormat.model_validate_json(prediction).model_dump()
                for prediction in predictions
            ]

    def find_index(self,lst,target_item):
        for  i , item in enumerate(lst):
            if target_item == item:
                return i

        return -1


    def find_index_in2(self, lst,target_item , mode = 'index'):
        for  i , item in enumerate(lst):
            # # print('target' ,target_item)
            # # print('item' ,item)
            # # print()
            if item    in target_item:

                # print('return')
                # print(item)
                # print(target_item)
                return i if mode =='index' else item

        return -1

    def find_index_in(self, lst,target_item , mode = 'index'):
        for  i , item in enumerate(lst):
            # # print('target' ,target_item)
            # # print('item' ,item)
            # # print()
            if target_item   in item:

                # print('return')
                # print(item)
                # print(target_item)
                return i if mode =='index' else target_item

        return -1


    def find_number(self,txt:list , mode = '=='):

        # locations = {}
        func = self.find_index if mode == '==' else self.find_index_in
        # locations = {}
        min_index = float('inf')
        min_number = ''
        for number in self.numbers:
            location    = func(txt , number)
            if location != -1:
                if location < min_index:
                    min_index = location
                    min_number = number



        # if mode != '==':
            # print('min index',min_index )
            # print('min_number' , min_number)



        return min_index , min_number



    def extract_headings(self,txt):
        txt = " ".join(txt.split()).replace('.' ,'').replace(',','').lower()
        splitted_text = txt.split(' ')


        while True:

            numbers =[]


            first_index , first_number = self.find_number(splitted_text)
            # try:
            if len(first_number) ==0:
                # breakpoint()
                splitted_text = txt.split(' ')
                # print(splitted_text)
                first_index , first_number = self.find_number(splitted_text, mode ='in')
                # # print('reset')
                # continue
            # except KeyError as e:
            numbers.append(self.number_words[first_number])
                # # print(e)
                # # print(txt)



            # second_index , third_index = first_index + 1, second_index+2
            try:
                for increment in range(1,3):
                    # # print(txt)
                    next_word = splitted_text[first_index + increment]
                    # print(next_word)
                    if next_word not in self.number_words.keys():
                        # breakpoint()

                        w = self.find_index_in2(self.number_words.keys() , next_word, mode = 'item')
                        if w != -1:
                            next_word = w
                    numbers.append(self.number_words[next_word])

                break
            except KeyError as e:
                print(e)
                print(txt)
                splitted_text= splitted_text[first_index + increment:]




                continue


        return "".join(numbers)


    def extract_info(self,transcript):

        pattern = re.compile(r'\b(' + '|'.join(self.start_target) + r').*?(' + '|'.join(self.end_target) + r')\b', re.IGNORECASE)
        match = pattern.search(transcript)
        if match:
            target = match.group()
        lowest_idx = float('inf')


        try:
            heading = self.extract_headings(transcript)
        except Exception as e:
            print(e)
            heading = '011'

        for t in self.all_tools:
            if t.lower() in transcript.lower():
                test = transcript.lower().find(t.lower())

                if transcript[test - 1] == '(':
                    tool = t
                    break
                if test < lowest_idx:
                    tool = t
                    lowest_idx = test

        if target.endswith('s'):
            target = target.replace('s', '')
        return {"heading":heading, "tool": tool, "target": target.lower()}