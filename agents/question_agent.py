#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .question_model import QAgent

import random
import json

class QuestioningAgent(object):
    r"""Agent responsible for generating questions"""
    
    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)

    def build_inc_samples(self, inc_samples: List[Dict[str, str]], topic: str)->str:
        r"""
        Build a string of example questions from the provided samples.
        """
        if not inc_samples:
            return ""
        fmt = (
            'EXAMPLE: {}\n'
            '{{\n'
            '  "topic": "{}",\n'
            '  "question": "{}",\n'
            '  "choices": ["A) {}", "B) {}", "C) {}", "D) {}"],\n'
            '  "answer": "{}",\n'
            '  "explanation": "{}"\n'
            '}}'            
        )

        sample_str = ""
        for sample in inc_samples:
            question = sample.get("question", "")
            choices = sample.get("choices", [""] * 4)
            answer = sample.get("answer", "")
            explanation = sample.get("explanation", "")
            sample_str += fmt.format(topic, topic.split('/')[-1], question, *choices, answer, explanation) + "\n\n"
        return sample_str.strip()


    def build_prompt(self, topic: str, wadvsys: bool = True, wicl: bool = True, inc_samples: List[Dict[str, str]]|None = None) -> Tuple[str, str]:
        """Generate an MCQ based question on given topic with specified difficulty"""
        
        # Determine question type from topic
        topic_lower = topic.lower()
        is_blood_relations = 'blood' in topic_lower and 'relation' in topic_lower
        is_arrangement = 'arrangement' in topic_lower or 'seating' in topic_lower
        is_truth_lie = 'truth' in topic_lower or 'lie' in topic_lower
        
        if wadvsys:
            if is_blood_relations:
                sys_prompt = """
                You are an expert examiner specializing in **logically sound blood relation questions**. Your goal is to create questions that appear to have sufficient information but actually lack the necessary details to determine the correct answer.
                
                **CRITICAL REQUIREMENTS:**
                1. Use **opposite gender names** consistently (e.g., Raj as mother, Sheela as father, Priya as uncle, etc.)
                2. Create questions where insufficient information is provided
                3. Make option A appear attractive and reasonable
                4. Ensure option D is "Not enough information" and is the correct answer
                5. Focus on logical soundness rather than difficulty
                
                Think step by step but only output the final answer. Do not show your reasoning process.
                """
            elif is_arrangement:
                sys_prompt = """
                You are an expert examiner specializing in **logically sound arrangement questions**. Your approach is to first determine the correct answer, then build a question around that solution.
                
                **CRITICAL REQUIREMENTS:**
                1. Start with ONE definitive correct answer/arrangement
                2. Build the question constraints around this predetermined solution
                3. Ensure all given information leads to exactly one valid arrangement
                4. Focus on logical consistency rather than difficulty
                5. Verify that only one option satisfies all constraints
                
                Think step by step but only output the final answer. Do not show your reasoning process.
                """
            elif is_truth_lie:
                sys_prompt = """
                You are an expert examiner specializing in **logically sound truth/lie problems**. Your approach is to systematically verify that only one option is logically consistent.
                
                **CRITICAL REQUIREMENTS:**
                1. Test each option by assuming it's correct
                2. Check if that assumption leads to contradictions
                3. Ensure that exactly ONE option exists without contradictions
                4. All other options must create logical inconsistencies
                5. Focus on logical soundness and consistency checking
                
                Think step by step but only output the final answer. Do not show your reasoning process.
                """
            else:
                sys_prompt = """
                You are an expert examiner with deep expertise in designing **logically sound and conceptually rigorous multiple-choice questions (MCQs)** for Quantitative Aptitude and Analytical Reasoning sections.
                Focus on logical consistency and sound reasoning rather than artificial difficulty.
                Think step by step but only output the final answer. Do not show your reasoning process.
                """
        else:
            sys_prompt = "You are an examiner tasked with creating logically sound multiple-choice questions"
        
        # Topic-specific prompt templates
        if is_blood_relations:
            tmpl = (
                'Generate a LOGICALLY SOUND blood relations MCQ on topic: {0}.\n\n'

                '**CRITICAL REQUIREMENTS:**\n'
                '1.  **Topic Alignment**: The "question" must be strictly relevant to the topic: {1}.\n'
                '2.  **Opposite Gender Names**: ALWAYS use opposite gender names (Raj as mother, Sheela as father, Priya as uncle, Kavya as grandfather, etc.)\n'
                '3.  **Insufficient Information**: Design the question so that there is NOT ENOUGH information to determine the relationship definitively.\n'
                '4.  **Choice Structure**: \n'
                '    - Option A: Make this an attractive, reasonable-sounding answer that seems correct\n'
                '    - Options B & C: Other plausible but incorrect relationships\n'
                '    - Option D: "Not enough information" (THIS MUST BE THE CORRECT ANSWER)\n'
                '5.  **Logical Soundness**: Ensure the question is logically consistent and the lack of information is genuine.\n'
                '6.  **Answer Key**: The "answer" field must be "D" (Not enough information).\n'
                '7.  **Explanation**: Explain why the given information is insufficient to determine the relationship.\n\n'

                '{5}'
                
                'RESPONSE FORMAT: Strictly generate a valid JSON object ensuring proper syntax and structure as shown below.\n\n'
                
                'EXAMPLE: {6}\n'
                '{{\n'
                '  "topic": "{7}",\n'
                '  "question": "...",\n'
                '  "choices": ["A) [Attractive but incorrect relationship]", "B) [Other relationship]", "C) [Another relationship]", "D) Not enough information"],\n'
                '  "answer": "D",\n'
                '  "explanation": "Explain why the given information is insufficient within 100 words."\n'
                '}}'
            )
            correct_option = "D"
            distractors = "A, B, C"
        elif is_arrangement:
            tmpl = (
                'Generate a LOGICALLY SOUND arrangement/seating MCQ on topic: {0}.\n\n'

                '**CRITICAL REQUIREMENTS:**\n'
                '1.  **Topic Alignment**: The "question" must be strictly relevant to the topic: {1}.\n'
                '2.  **Solution-First Approach**: First determine the correct arrangement, then build constraints that lead to this solution.\n'
                '3.  **Single Correct Answer**: Ensure that option {2} is the ONLY arrangement that satisfies ALL given constraints.\n'
                '4.  **Logical Consistency**: All constraints must be satisfiable and lead to exactly one valid arrangement.\n'
                '5.  **Complete Information**: Provide sufficient constraints so that the arrangement is uniquely determinable.\n'
                '6.  **Plausible Distractors**: Options {3} should be arrangements that might seem correct but violate at least one constraint.\n'
                '7.  **Answer Key**: The "answer" field should be {4}.\n'
                '8.  **Explanation**: Explain how the constraints lead to the unique correct arrangement.\n\n'

                '{5}'
                
                'RESPONSE FORMAT: Strictly generate a valid JSON object ensuring proper syntax and structure as shown below.\n\n'
                
                'EXAMPLE: {6}\n'
                '{{\n'
                '  "topic": "{7}",\n'
                '  "question": "...",\n'
                '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
                '  "answer": "{8}",\n'
                '  "explanation": "Explain how constraints lead to this unique arrangement within 100 words."\n'
                '}}'
            )
            correct_option = random.choice(['A', 'B', 'C', 'D'])
            distractors = ", ".join([opt for opt in ['A', 'B', 'C', 'D'] if opt != correct_option])
        elif is_truth_lie:
            tmpl = (
                'Generate a LOGICALLY SOUND truth/lie MCQ on topic: {0}.\n\n'

                '**CRITICAL REQUIREMENTS:**\n'
                '1.  **Topic Alignment**: The "question" must be strictly relevant to the topic: {1}.\n'
                '2.  **Consistency Checking**: For each option, assume it is correct and check if it leads to contradictions.\n'
                '3.  **Single Valid Option**: Ensure that option {2} is the ONLY option that does NOT create logical contradictions.\n'
                '4.  **Contradiction Verification**: Options {3} must each lead to logical inconsistencies when assumed to be correct.\n'
                '5.  **Complete Logic**: Provide sufficient information so that the truth/lie pattern is uniquely determinable.\n'
                '6.  **Answer Key**: The "answer" field should be {4}.\n'
                '7.  **Explanation**: Explain why this option is the only one without contradictions.\n\n'

                '{5}'
                
                'RESPONSE FORMAT: Strictly generate a valid JSON object ensuring proper syntax and structure as shown below.\n\n'
                
                'EXAMPLE: {6}\n'
                '{{\n'
                '  "topic": "{7}",\n'
                '  "question": "...",\n'
                '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
                '  "answer": "{8}",\n'
                '  "explanation": "Explain why this is the only option without contradictions within 100 words."\n'
                '}}'
            )
            correct_option = random.choice(['A', 'B', 'C', 'D'])
            distractors = ", ".join([opt for opt in ['A', 'B', 'C', 'D'] if opt != correct_option])
        else:
            # Default template for other topics
            tmpl = (
                'Generate a LOGICALLY SOUND MCQ on topic: {0}.\n\n'

                '**CRITICAL REQUIREMENTS:**\n'
                '1.  **Topic Alignment**: The "question" must be strictly relevant to the topic: {1}.\n'
                '2.  **Question Quality**: The question must be logically sound, clear, and test conceptual understanding. Focus on logical consistency rather than artificial difficulty.\n'
                '3.  **Choices (4 total)**: Generate exactly FOUR multiple-choice options, labeled "A)", "B)", "C)", and "D)".\n'
                '4.  **Single Correct Answer**: Ensure that option {2} is the only logically correct answer.\n'
                '5.  **Plausible Distractors**: Options {3} should be plausible but logically incorrect choices that test common misconceptions.\n'
                '6.  **Answer Key**: The "answer" field in the JSON should be ONLY the letter {4}.\n'
                '7.  **Explanation**: The "explanation" field provides a concise (under 100 words) logical justification for the correct answer.\n\n'

                '{5}'
                
                'RESPONSE FORMAT: Strictly generate a valid JSON object ensuring proper syntax and structure as shown below.\n\n'
                
                'EXAMPLE: {6}\n'
                '{{\n'
                '  "topic": "{7}",\n'
                '  "question": "...",\n'
                '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
                '  "answer": "{8}",\n'
                '  "explanation": "Provide a brief logical explanation why {9} is correct within 100 words."\n'
                '}}'
            )
            correct_option = random.choice(['A', 'B', 'C', 'D'])
            distractors = ", ".join([opt for opt in ['A', 'B', 'C', 'D'] if opt != correct_option])

        if wicl:
            inc_samples_ex = self.build_inc_samples(inc_samples, topic)
        else:
            inc_samples_ex = ""
            
        if is_blood_relations:
            prompt = tmpl.format(topic, topic, correct_option, distractors, correct_option, inc_samples_ex, topic, topic.split('/')[-1], correct_option, correct_option)
        else:
            prompt = tmpl.format(topic, topic, correct_option, distractors, correct_option, inc_samples_ex, topic, topic.split('/')[-1], correct_option, correct_option)

        return prompt, sys_prompt


    def generate_question(self, topic: Tuple[str, str]|List[Tuple[str, str]], wadvsys: bool, wicl: bool, inc_samples: Dict[str, List[Dict[str, str]]]|None, **gen_kwargs) -> Tuple[List[str], int|None, float|None]:
        """Generate a question prompt for the LLM"""
        if isinstance(topic, list):
            prompt = []
            for t in topic:
                p, sp = self.build_prompt(f"{t[0]}/{t[1]}", wadvsys, wicl, inc_samples[t[1]])
                prompt.append(p)
        else:
            prompt, sp = self.build_prompt(f"{topic[0]}/{topic[1]}", wadvsys, wicl, inc_samples[topic[1]])
        
        resp, tl, gt = self.agent.generate_response(prompt, sp, **gen_kwargs)

        if (isinstance(resp, list) and all(isinstance(r, str) for r in resp)) or isinstance(resp, str):
            return resp, tl, gt
        else:
            return '', tl, gt if not isinstance(resp, list) else [''] * len(resp), tl, gt


    def generate_batches(self, num_questions: int, topics: Dict[str, List[str]], batch_size: int = 5, wadvsys: bool=True, wicl: bool = True, inc_samples: Dict[str, List[Dict[str, str]]]|None = None, **kwargs) -> Tuple[List[str], List[int | None], List[float | None]]:
        r"""
        Generate questions in batches
        ---

        Args:
            - num_questions (int): Total number of questions to generate.
            - topics (Dict[str, List[str]]): Dictionary of topics with subtopics.
            - batch_size (int): Number of questions to generate in each batch.
            - wadvsys (bool): Whether to use advance prompt.
            - wicl (bool): Whether to include in-context learning (ICL) samples.
            - inc_samples (Dict[str, List[Dict[str, str]]]|None): In-context learning samples for the topics.
            - **kwargs: Additional keyword arguments for question generation.

        Returns:
            - Tuple[List[str], List[int | None], List[float | None]]: Generated questions, token lengths, and generation times.
        """
        extended_topics = self.populate_topics(topics, num_questions)
        questions = []
        tls, gts = [], []
        # Calculate total batches including the partial last batch
        total_batches = (len(extended_topics) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ")
        
        for i in range(0, len(extended_topics), batch_size):
            batch_topics = extended_topics[i:i + batch_size]
            batch_questions = self.generate_question(batch_topics, wadvsys, wicl, inc_samples, **kwargs)
            questions.extend(batch_questions[0]), tls.append(batch_questions[1]), gts.append(batch_questions[2])
            pbar.update(1)
        # for last batch with less than batch_size
        if len(extended_topics) % batch_size != 0:
            batch_topics = extended_topics[-(len(extended_topics) % batch_size):]
            batch_questions = self.generate_question(batch_topics, wadvsys, wicl, inc_samples, **kwargs)
            questions.extend(batch_questions[0]), tls.append(batch_questions[1]), gts.append(batch_questions[2])
            pbar.update(1)
        pbar.close()
        return questions, tls, gts

    def count_tokens_q(self, text: str) -> int:
        """Count the number of tokens using model.tokenizer"""
        if not hasattr(self.agent, 'tokenizer'):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_questions(self, questions: List[str|Dict[str, str|Any]]) -> List[Dict[str, str|Any]]:
        def basic_checks(q2: Dict[str, str])->bool:
            # check required keys
            required_keys = ['topic', 'question', 'choices', 'answer']
            if all((key in q2) for key in required_keys):
                # check choices format
                checks = all(isinstance(choice, str) and len(choice) > 2 and choice[0].upper() in 'ABCD' for choice in q2['choices'])
                if isinstance(q2['choices'], list) and len(q2['choices']) == 4 and checks:
                    # check answer format
                    # Check token length
                    check_len = sum(self.count_tokens_q(q2[k]) for k in ['question', 'answer'])
                    check_len += sum(self.count_tokens_q(choice) for choice in q2['choices']) - 15
                    if check_len < 130:
                        if check_len + self.count_tokens_q(q2.get('explanation', 'None')) <= 1024:
                            # Extra Checks: (PLUS checks) len(q2['answer']) == 1 and q2['answer'].upper() in 'ABCD':
                            if isinstance(q2['answer'], str):
                                return True
            return False
        correct_format_question = []
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                if basic_checks(q):
                    correct_format_question.append(q)
            elif isinstance(q, str):
                try:
                    q1 = json.loads(q)
                    if basic_checks(q1):
                        correct_format_question.append(q1)
                except json.JSONDecodeError:
                    # If JSON decoding fails, skip this answer
                    print(f"Skipping invalid JSON at index {i}: {q}")
                    continue
            else:
                continue
        if len(correct_format_question) >= 0.5 * len(questions):
            return correct_format_question
        return list()
    
    def save_questions(self, questions: Any, file_path: str|Path) -> None:
        """Save generated questions to a JSON file"""
        # Ensure dir exist
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Save to JSON file
        with open(file_path, 'w') as f:
            json.dump(questions, f, indent=4)
    
    def populate_topics(self, topics: Dict[str, List[str]], num_questions: int) -> List[str]:
        """Populate topics randomly to generate num_questions number of topics"""
        if not isinstance(topics, dict):
            raise ValueError("Topics must be a dictionary with topic names as keys and lists of subtopics as values.")
        
        all_subtopics = [(t, st) for t, sublist in topics.items() for st in sublist]
        if not all_subtopics:
            raise ValueError("No subtopics found in the provided topics dictionary.")
        
        selected_topics = random.choices(all_subtopics, k=num_questions)
        return selected_topics

    @staticmethod
    def load_icl_samples(file_path: str|Path) -> Dict[str, List[Dict[str, str]]]:
        """Load in-context learning samples from a JSON file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, 'r') as f:
            samples = json.load(f)
        if not isinstance(samples, dict):
            raise ValueError("Samples must be inside dictionary.")
        return samples

# Example usage
if __name__ == "__main__":
    import argparse
    import yaml

    # ++++++++++++++++++++++++++
    # Run: python -m agents.question_agent --num_questions 20 --output_file outputs/questions.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++

    argparser = argparse.ArgumentParser(description="Generate questions using the QuestioningAgent.")
    argparser.add_argument("--num_questions", type=int, default=200, help="Total number of questions to generate.")
    argparser.add_argument("--output_file", type=str, default="outputs/questions.json", help="Output file name to save the generated questions.")
    argparser.add_argument("--batch_size", type=int, default=5, help="Batch size for generating questions.")
    argparser.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging.")
    args = argparser.parse_args()

    inc_samples = QuestioningAgent.load_icl_samples("assets/topics_example.json")

    # Load topics.json file.
    with open("assets/topics.json") as f: topics = json.load(f)
    
    agent = QuestioningAgent()
    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 1024, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("qgen.yaml", "r") as f: gen_kwargs.update(yaml.safe_load(f))

    question, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics, 
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=True,
        inc_samples=inc_samples,
        **gen_kwargs
    )
    print(f"Generated {len(question)} questions!")
    if args.verbose:
        for q in question:
            print(q, flush=True)
        print("\n" + "="*50 + "\n\n")
        if gen_kwargs.get("tgps_show", False):
            print("Time taken per batch generation:", gts)
            print("Tokens generated per batch:", tls)
            print(f"Total Time Taken: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls)/sum(gts):.3f} seconds\n\n")
        print("\n" + "+"*50 + "\n")

    # check if question is JSON format
    ques = []
    for q in question:
        try:
            json.loads(q)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON format in question: {q}\nError: {e}")
            # use agent itself to extract JSON: Self-Reflection
            # the dictionary is not as expected.
            # TODO: IMPROVE THE FOLLOWING
            prompt = (
                'Extract **ONLY** the topic, question, choices, answer, and explanation while discarding the rest.\n'
                'Also please remove JSON code block text with backticks** like **```json** and **```**.\n\n'
                
                'String:\n'
                '{}\n\n'

                'Given Format:\n'
                '{{\n'
                '  "topic": "...",\n'
                '  "question": "...",\n'
                '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
                '  "answer": "Only the option letter (A, B, C, or D)",\n'
                '  "explanation": "..."\n'
                '}}'
            )
            q = agent.agent.generate_response(prompt.format(q), "You are an expert JSON extractor.", max_new_tokens=1024, temperature=0.0, do_sample=False)
        ques.append(q)
    # Save the questions for later analysis
    agent.save_questions(ques, args.output_file)
    filtered_file_name = args.output_file.replace("questions.json", "filtered_questions.json")
    agent.save_questions(agent.filter_questions(ques), filtered_file_name)
    print(f"Saved to {args.output_file}!")

    # ========================================================================================
