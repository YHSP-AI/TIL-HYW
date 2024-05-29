from typing import Dict
import json
import re






class NLPManager:
    def __init__(self):
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

    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering

        return self.extract_info(context)


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
        func = self.all_toolsfind_index if mode == '==' else self.all_toolsfind_index_in
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
                breakpoint()
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
                        breakpoint()

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
        except:
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