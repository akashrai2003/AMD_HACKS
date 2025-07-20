#!/usr/bin/python3

import re
import json

from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from .answer_model_2 import AAgent

class AnsweringAgent(object):
    r"""Agent responsible for answering MCQ questions with confidence scoring"""
    
    def __init__(self, select_prompt1: bool = True, **kwargs):
        self.agent = AAgent(**kwargs)
        self.select_prompt1 = select_prompt1
    
    def build_prompt(self, question_data: Dict[str, str|Any]) -> Tuple[str, str]:
        """Generate an answer to the given MCQ question with confidence and reasoning"""
        
        sys_prompt1 = """You are an MCQ answering model.
When you look at the problems, I want you to create a JSON structure which has keys "question_type", "intermediate_steps", and "answer". 
The question type must strictly be from "blood_relations"/"seating_arrangement"/"truth_lie".
The intermediate step depends on your question type.
The answer must be from one of "A","B","C","D"
The full output should be under 512 tokens.

The "intermediate_steps" value must follow this JSON structure as shown in these 3 examples. The structure will be based on the "question_type" key, and must follow the pattern strictly.
Then it must be populated by your logic and reasoning.

Given, 3 examples of the full output based on the 3 different "question_types":

EXAMPLE 1 (blood_relations):
Input: 
Question: Raj is the parent of Sam. Sam, who is the sibling of Priya, later becomes the parent of Tina. Priya has no children. Based on the given information, how is Priya related to Tina?
Choices:
A) Aunt
B) Grandmother
C) Sister
D) Mother

Output:
{
  "question_type": "blood_relations",
  "intermediate_steps": {
    "people_order": [
      "Raj",
      "Sam", 
      "Priya",
      "Tina"
    ],
    "people": {
      "Raj": {
        "gender": "",
        "relations": {
          "child": [
            "Sam"
          ]
        }
      },
      "Sam": {
        "gender": "",
        "relations": {
          "parent": "Raj",
          "sibling": [
            "Priya"
          ],
          "child": [
            "Tina"
          ]
        }
      },
      "Priya": {
        "gender": "",
        "relations": {
          "sibling": [
            "Sam"
          ]
        }
      },
      "Tina": {
        "gender": "",
        "relations": {
          "parent": "Sam",
          "aunt": [
            "Priya"
          ]
        }
      }
    },
    "question_repeat": "Raj is the parent of Sam. Sam, who is the sibling of Priya, later becomes the parent of Tina. Priya has no children. Based on the given information, how is Priya related to Tina?",
    "options_repeat": {
      "A": "Aunt",
      "B": "Grandmother", 
      "C": "Sister",
      "D": "Mother"
    }
  },
  "answer": "A"
}

EXAMPLE 2 (seating_arrangement):
Input:
Question: Four people, John, Sarah, Mike, and Emily, are sitting in a straight line. John is seated immediately to the right of Sarah, who faces north. Mike is seated immediately to the left of Emily, who also faces north. If John faces south, who is seated immediately to the left of Mike?
Choices:
A) John
B) Sarah
C) Emily  
D) No one

Output:
{
  "question_type": "seating_arrangement",
  "intermediate_steps": {
    "arrangement": "linear",
    "people": [
      "John",
      "Sarah",
      "Mike", 
      "Emily"
    ],
    "facing": {
      "John": "south",
      "Sarah": "north",
      "Mike": "north",
      "Emily": "north"
    },
    "relations": [
      {
        "type": "immediate_right",
        "subject": "John",
        "reference": "Sarah"
      },
      {
        "type": "immediate_left", 
        "subject": "Mike",
        "reference": "Emily"
      }
    ],
    "question_repeat": "Four people, John, Sarah, Mike, and Emily, are sitting in a straight line. John is seated immediately to the right of Sarah, who faces north. Mike is seated immediately to the left of Emily, who also faces north. If John faces south, who is seated immediately to the left of Mike?",
    "options_repeat": {
      "A": "John",
      "B": "Sarah",
      "C": "Emily",
      "D": "No one"
    }
  },
  "answer": "B"
}

EXAMPLE 3 (truth_lie):
Input:
Question: Four friends, X, Y, Z, and W, are discussing who among them is telling the truth. X says: 'Y and Z are both lying.' Y responds by saying: 'X and W are both telling the truth.' Z then claims: 'X and Y are both lying.' W states: 'Y is lying and Z is telling the truth.' Based on the statements made by X, Y, Z, and W, and knowing that exactly two of them are telling the truth, determine who the truthful persons are.
Choices:
A) X and Y
B) Y and Z  
C) X and W
D) Y and W

Output:
{
  "question_type": "truth_lie",
  "intermediate_steps": {
    "people": [
      "X",
      "Y", 
      "Z",
      "W"
    ],
    "assumptions": {
      "A": {
        "assume": {
          "X": "T",
          "Y": "T",
          "Z": "F",
          "W": "F"
        },
        "check": [
          "X ok",
          "Y ok", 
          "Z x",
          "W x"
        ]
      },
      "B": {
        "assume": {
          "X": "F",
          "Y": "T",
          "Z": "T",
          "W": "F"
        },
        "check": [
          "X x",
          "Y ok",
          "Z ok", 
          "W x"
        ]
      },
      "C": {
        "assume": {
          "X": "T",
          "Y": "F",
          "Z": "F",
          "W": "T"
        },
        "check": [
          "X ok",
          "Y x",
          "Z x",
          "W ok"
        ]
      },
      "D": {
        "assume": {
          "X": "F",
          "Y": "T", 
          "Z": "F",
          "W": "T"
        },
        "check": [
          "X x",
          "Y ok",
          "Z x",
          "W ok"
        ]
      }
    },
    "question_repeat": "Four friends, X, Y, Z, and W, are discussing who among them is telling the truth. X says: 'Y and Z are both lying.' Y responds by saying: 'X and W are both telling the truth.' Z then claims: 'X and Y are both lying.' W states: 'Y is lying and Z is telling the truth.' Based on the statements made by X, Y, Z, and W, and knowing that exactly two of them are telling the truth, determine who the truthful persons are.",
    "options_repeat": {
      "A": "X and Y",
      "B": "Y and Z",
      "C": "X and W", 
      "D": "Y and W"
    }
  },
  "answer": "C"
}"""
        sys_prompt2 = sys_prompt1  # Both system prompts are the same now

        
        tmpl = (
            'Now answer the following question:\n'
            'Question: {}\n'
            'Choices: {}\n\n'
        )
        
        prompt = tmpl.format(
            question_data['question'],
            self._format_choices(question_data['choices'])
        )
        
        return prompt, sys_prompt1 if self.select_prompt1 else sys_prompt2
    
    def answer_question(self, question_data: Dict|List[Dict], **kwargs) -> Tuple[List[Dict], int|None, float|None]:
        """Generate answer(s) for the given question(s)"""
        if isinstance(question_data, list):
            prompt = []
            for qd in question_data:
                p, sp = self.build_prompt(qd)
                prompt.append(p)
        else:
            prompt, sp = self.build_prompt(question_data)
        
        resp, tl, gt = self.agent.generate_response(prompt, sp, **kwargs)

        if (isinstance(resp, list) and all(isinstance(r, str) for r in resp)) or isinstance(resp, str):
            return resp, tl, gt
        else:
            return '', tl, gt if not isinstance(resp, list) else [''] * len(resp), tl, gt
    
    def answer_batches(self, questions: List[Dict], batch_size: int = 5, **kwargs) -> Tuple[List[Dict], List[int | None], List[float | None]]:
        """Answer questions in batches"""
        answers = []
        tls, gts = [], []
        total_batches = (len(questions) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ", unit="batch")
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_answers, tl, gt = self.answer_question(batch_questions, **kwargs)
            answers.extend(batch_answers)
            tls.append(tl); gts.append(gt)
            pbar.update(1)
        
        # Handle last batch with less than batch_size
        if len(questions) % batch_size != 0:
            batch_questions = questions[-(len(questions) % batch_size):]
            batch_answers = self.answer_question(batch_questions, **kwargs)
            answers.extend(batch_answers[0]); tls.append(batch_answers[1]); gts.append(batch_answers[2])
            pbar.update(1)
        pbar.close()
        return answers, tls, gts
    
    def count_tokens_a(self, text: str) -> int:
        """Count the number of tokens in the text using the agent's tokenizer"""
        if not hasattr(self.agent, 'tokenizer'):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_answers(self, ans: List[str|Dict[str, str]]) -> List[Dict[str, str]]:
        r"""Filter answers to ensure they are in the correct format"""
        def basic_checks(a1: Dict[str, str])->bool:
            # check required keys
            required_keys = ['answer']
            if all((key in a1) and isinstance(a1[key], str) for key in required_keys):
                if len(a1['answer']) == 1 and (a1['answer'] not in 'ABCDabcd'):
                    return False
                check_len = self.count_tokens_a(a1['answer'])
                if check_len < 50:
                    check_len += self.count_tokens_a(a1.get('reasoning', 'None'))
                    if check_len < 512:
                        # check answer format - EXTRA checks
                        # if len(a1['answer']) == 1 and a1['answer'].upper() in 'ABCD':
                        return True
            return False
    
        filtered_answers = []
        for i, a in enumerate(ans):
            if isinstance(a, dict):
                if basic_checks(a):
                    filtered_answers.append(a)
                else:
                    filtered_answers.append(None)
                    print(f"Skipping invalid answer at index {i}: {a}")
            elif isinstance(a, str):
                # Basic checks: at least with correct JSON format
                try:
                    a1 = json.loads(a)
                    if basic_checks(a1):
                        filtered_answers.append(a1)
                    else:
                        filtered_answers.append(None)
                        print(f"Skipping invalid answer at index {i}: {a}")
                except json.JSONDecodeError:
                    # If JSON decoding fails, skip this answer
                    print(f"Skipping invalid JSON at index {i}: {a}")
                    filtered_answers.append(None)
                    continue
            else:
                # If the answer is neither a dict nor a str, skip it
                print(f"Skipping unsupported type at index {i}: {type(a)}")
                filtered_answers.append(None)
        return filtered_answers

    def save_answers(self, answers: List[str], file_path: str|Path) -> None:
        """Save generated answers to a JSON file"""
        # check for existence of dir
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump([a for a in answers], f, indent=4)
    
    def _format_choices(self, choices: List[str]) -> str:
        r"""Format the choices for better readability"""
        formatted = []
        for choice in choices:
            # Ensure each choice starts with a letter if not already formatted
            if not re.match(r'^[A-D]\)', choice.strip()):
                # Extract letter from existing format or assign based on position
                letter = chr(65 + len(formatted))  # A, B, C, D
                formatted.append(f"{letter}) {choice.strip()}")
            else:
                formatted.append(choice.strip())
        return " ".join(formatted)


# Example usage
if __name__ == "__main__":
    import json
    import yaml
    import argparse
    from utils.build_prompt import auto_json, option_extractor_prompt
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # python -m agents.answer_agent_2 --input_file outputs/filtered_questions.json --output_file outputs/answers.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    argparser = argparse.ArgumentParser(description="Run the Answering Agent")
    argparser.add_argument("--input_file", type=str, default="outputs/filtered_questions.json", help="Path to the input JSON file with questions")
    argparser.add_argument("--output_file", type=str, default="outputs/answers.json", help="Path to save the answers")
    argparser.add_argument("--batch_size", type=int, default=5, help="Batch size for processing questions")
    argparser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    args = argparser.parse_args()

    SELECT_PROMPT1 = False  # Use the first system prompt for answering
    
    # Load sample questions (assuming they're saved from QuestioningAgent)
    with open(args.input_file, 'r') as f:
        sample_questions = json.load(f)
    
    agent = AnsweringAgent(select_prompt1=SELECT_PROMPT1, adapter_type="sft")
    
    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 512, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("agen.yaml", "r") as f: gen_kwargs.update(yaml.safe_load(f))
    answer, tls, gts = agent.answer_batches(
        questions=sample_questions,
        batch_size=args.batch_size,
        **gen_kwargs
    )
    ans = []
    for idx, (q, a) in enumerate(zip(sample_questions, answer)):
        if args.verbose:
            print(f"\n=== Question {idx+1} ===")
            print(f"Question: {q.get('question', 'N/A')}")
            print(f"Expected: {q.get('answer', 'N/A')}")
            print(f"Model Answer:\n{a}")
        try:
            a = json.loads(a)
            if all(k in a for k in ['answer', 'reasoning']):
                # ++++++++++++++++++++++++++
                # TODO: IMPROVE THE FOLLOWING
                if len(a['answer']) != 1:
                    a['answer'] = agent.agent.generate_response(option_extractor_prompt(a['answer'], q['choices']))
                # ++++++++++++++++++++++++++
            else:
                # the dictionary is not as expected. So extract it using the same model: Self-Reflection
                prompt = (
                    'Extract **ONLY** the answer and reasoning while discarding the rest.\n\n'
                    
                    'String:\n'
                    '{}\n\n'

                    'Given Format:\n'
                    '{{\n'
                    '    "answer": "Only the option letter (A, B, C, or D)",\n'
                    '    "reasoning": "..."\n'
                    '}}'
                )
                a = agent.agent.generate_response(prompt.format(json.dumps(a, indent=4)))
        except json.JSONDecodeError:
            a = agent.agent.generate_response(auto_json(a))
        ans.append(a)
        
    if args.verbose:
        if gen_kwargs.get('tgps_show', False):
            for idx, (tl, gt) in enumerate(zip(tls, gts)):
                print(f"BATCH - {idx}")
                print(f"Tokens: {tl}, Time: {gt:.3f} seconds")
                print(f"TGPS: {tl/gt:.3f} seconds")
            print("\n" + "="*50)
            print(f"Total Time: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls)/sum(gts):.3f} seconds")
    
    # Save answers
    agent.save_answers(ans, args.output_file)
    filtered_file_name = args.output_file.replace("answers.json", "filtered_answers.json")
    agent.save_answers(agent.filter_answers(ans), filtered_file_name) 