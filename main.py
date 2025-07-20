#!/usr/bin/env python3
"""
🏆 HACKATHON COMPETITION CLI TOOL 🏆
Fixed version with JSON parsing and no thinking
"""

import os
import json
import time
import torch
import re
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from datetime import datetime
import random 

class CompetitionAgent:
    """Unified agent for both question generation and answer solving"""
    
    def __init__(self, agent_type: str, model_path: str = "/jupyter-tutorial/hf_models/Qwen3-4B"):
        self.agent_type = agent_type  # "question" or "answer"
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.token_limit = 100
        
        # Load appropriate checkpoint
        if agent_type == "question":
            self.checkpoint_dir = "checkpoints/question_agent_sft"
        else:
            self.checkpoint_dir = "checkpoints/answer_agent_sft"

        
        self.system_prompts = self._load_system_prompts()

    def _load_system_prompts(self) -> Dict[str, str]:
        """Load optimized system prompts"""
        if self.agent_type == "question":
            return {
                "default": """# Unified MCQ Generation System Prompt

You are an expert examiner with deep expertise in designing **logically sound and conceptually rigorous multiple-choice questions (MCQs)** for Quantitative Aptitude and Analytical Reasoning sections. Your primary focus is on logical consistency and sound reasoning rather than artificial difficulty.

## Core Principles
- **Logical Soundness**: Every question must be logically consistent with exactly one correct answer
- **Topic Alignment**: Questions must be strictly relevant to the specified topic
- **Complete Information**: Provide sufficient constraints for deterministic solutions (except for blood relations)
- **Quality over Complexity**: Focus on conceptual rigor rather than artificial difficulty

## Topic-Specific Requirements

### Blood Relations Questions
When generating blood relations questions:
1. **Use opposite gender names consistently** (e.g., Raj as mother, Sheela as father, Priya as uncle, Kavya as grandfather)
2. **Design for insufficient information** - create scenarios where there isn't enough data to determine relationships definitively
3. **Choice Structure**:
   - Option A: Attractive, reasonable-sounding but incorrect answer
   - Options B & C: Other plausible but incorrect relationships
   - Option D: "Not enough information" (ALWAYS the correct answer)
4. **Answer Key**: Must always be "D"
5. **Explanation**: Explain why the given information is insufficient

### Arrangement/Seating Questions
When generating arrangement questions:
1. **Solution-First Approach**: Determine the correct arrangement first, then build constraints around it
2. **Single Correct Answer**: Ensure only one option satisfies ALL given constraints
3. **Complete Information**: Provide sufficient constraints for unique determination
4. **Logical Consistency**: All constraints must be satisfiable simultaneously
5. **Plausible Distractors**: Other options should seem correct but violate at least one constraint
6. **Answer Key**: Can be any option (A, B, C, or D)
7. **Explanation**: Show how constraints lead to the unique correct arrangement

### Truth/Lie Questions
When generating truth/lie questions:
1. **Consistency Checking**: Test each option by assuming it's correct
2. **Contradiction Verification**: Ensure only one option avoids logical contradictions
3. **Complete Logic**: Provide sufficient information for unique determination
4. **Systematic Approach**: All other options must create logical inconsistencies
5. **Answer Key**: Can be any option (A, B, C, or D)
6. **Explanation**: Explain why this option is the only one without contradictions

### General Questions
For other topics:
- Focus on conceptual understanding and logical reasoning
- Ensure single correct answer with well-constructed distractors
- Maintain topic relevance and appropriate difficulty level

## Output Requirements
- **Think step by step internally but only output the final answer**
- **Do not show your reasoning process in the response**
- **Generate valid JSON format with proper syntax**
- **Keep explanations within 100 words**
- **Ensure all choices are clearly distinguishable**

## Response Format
Always respond with a valid JSON object:
```json
{
  "topic": "[specified topic]",
  "question": "[well-crafted question text]",
  "choices": ["A) [option]", "B) [option]", "C) [option]", "D) [option]"],
  "answer": "[correct option letter]",
  "explanation": "[clear explanation within 100 words]"
}
```

Remember: Your goal is to create questions that test genuine understanding and logical reasoning skills while maintaining perfect logical consistency."""
            }
        else:  # answer agent
            return {
                "default": """You are an MCQ answering model.
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
            }
    
    def load_model(self):
        """Load the trained model"""
        if self.model is not None:
            return  # Already loaded
        
        print(f"🚀 Loading {self.agent_type.title()} Agent...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load LoRA adapters if available
            if os.path.exists(self.checkpoint_dir):
                print(f"✅ Loading trained checkpoint from {self.checkpoint_dir}")
                self.model = PeftModel.from_pretrained(base_model, self.checkpoint_dir)
            else:
                print(f"⚠️  No checkpoint found at {self.checkpoint_dir}, using base model")
                self.model = base_model
            
            self.model.eval()
            print(f"✅ {self.agent_type.title()} Agent loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))


    def extract_json_response(self, response: str) -> Dict:
        """
        Parses LLM response for structured question-answer JSON format.
        Expects keys: topic, question, choices, answer, explanation.
        Returns a dict with those keys if available.
        """
        try:
            # Strip markdown formatting if present
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = response.strip()
    
            # Basic JSON repair
            if response.count('{') > response.count('}'):
                response += '}' * (response.count('{') - response.count('}'))
            response = re.sub(r',\s*([}\]])', r'\1', response)
    
            # Parse JSON
            data = json.loads(response)
    
            # Extract required fields
            topic = data.get("topic", "").strip()
            question = data.get("question", "").strip()
            choices = data.get("choices", [])
            answer = data.get("answer", "").strip().upper()
            explanation = data.get("explanation", "").strip()
    
            # Return structured response
            return {
                "topic": topic,
                "question": question,
                "choices": choices,
                "answer": answer,
                "explanation": explanation
            }
    
        except Exception as e:
            return {"error": f"Failed to parse JSON or extract fields: {str(e)}", "raw_response": response[:500]}

        
    def extract_json_from_response(self, response: str) -> Dict:
        """
        Extracts the 'answer' and generates a 'reasoning' string from a JSON response.
        Handles markdown/code formatting and minor JSON repair.
        """
        try:
            # Clean markdown/code blocks
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
    
            # Attempt to repair common JSON issues
            response = response.strip()
    
            # Try to add closing braces if missing
            if response.count('{') > response.count('}'):
                response += '}' * (response.count('{') - response.count('}'))
    
            # Remove trailing commas before closing brackets or braces
            response = re.sub(r',\s*([}\]])', r'\1', response)
    
            data = json.loads(response)
    
            answer = data.get("answer", "").strip().upper()
    
            # Generate reasoning from intermediate_steps
            reasoning_raw = data.get("intermediate_steps", {})
    
            def flatten(d, prefix=""):
                """Helper to flatten nested dicts/lists into readable text lines"""
                lines = []
                if isinstance(d, dict):
                    for k, v in d.items():
                        full_key = f"{prefix}{k}".replace("_", " ").capitalize()
                        if isinstance(v, (dict, list)):
                            lines.extend(flatten(v, prefix=full_key + ": "))
                        else:
                            lines.append(f"{full_key}: {v}")
                elif isinstance(d, list):
                    for item in d:
                        lines.extend(flatten(item, prefix=prefix))
                else:
                    lines.append(f"{prefix}{d}")
                return lines
    
            reasoning_lines = flatten(reasoning_raw)
            reasoning = " | ".join(reasoning_lines)
    
            return {
                "answer": answer,
                "reasoning": reasoning if reasoning else "No detailed reasoning found."
            }
    
        except Exception as e:
            return {"error": f"Failed to parse JSON or extract fields: {str(e)}", "raw_response": response[:500]}
    
        
    def validate_question_tokens(self, question_data: Dict) -> tuple[bool, int]:
        """Validate question meets token limit"""
        if self.agent_type != "question" or "error" in question_data:
            return True, 0
        
        # Count core tokens (excluding explanation)
        core_content = f"{question_data.get('topic', '')} {question_data.get('question', '')} {' '.join(question_data.get('choices', []))} {question_data.get('answer', '')}"
        token_count = self.count_tokens(core_content)
        
        return token_count <= self.token_limit, token_count
    
    def generate_question(self, difficulty: str = "championship") -> Dict:
        """Generate a competition-quality Number Series question"""
        if self.agent_type != "question":
            raise ValueError("This agent is not configured for question generation")
        
        self.load_model()
        
        # Use optimized prompt for Number Series
        system_prompt = self.system_prompts["default"]

        topics_str = ["truth_lie","blood_relations","seating_arrangement"]
        topic = random.choice(topics_str)
        
        user_prompt = f"""Create an extremely difficult question from the topic given below:{topic}
         """
        
        # Generate question
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._generate_response(messages, max_tokens=400)
        
        # Extract JSON from response
        # question_data = self.extract_json_from_response(response)
        question_data = self.extract_json_response(response)
        print(question_data)
        if "error" not in question_data:
            # Validate token limit
            is_valid, token_count = self.validate_question_tokens(question_data)
            question_data["token_count"] = token_count
            question_data["token_valid"] = is_valid
            
            if not is_valid:
                print(f"⚠️  Token limit exceeded: {token_count}/100")
        
        return question_data
    
    def solve_question(self, question_data: Dict) -> Dict:
        """Solve a given question"""
        if self.agent_type != "answer":
            raise ValueError("This agent is not configured for answer solving")
        
        self.load_model()
        
        # Get system prompt
        system_prompt = self.system_prompts["default"]
        
        # Format question for solving (matching answer_agent.py format)
        question_text = question_data.get("question", "")
        choices = question_data.get("choices", [])
        
        # Use the exact format from answer_agent.py
        user_prompt = f"""
Now answer the following question:
Question: {question_text}
Choices: {' '.join(choices)}
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._generate_response(messages, max_tokens=800)
        
        # Extract JSON from response
        answer_data = self.extract_json_from_response(response)
        return answer_data
    
    def _generate_response(self, messages: List[Dict], max_tokens: int = 400) -> str:
        """Generate response using the model"""
        # Apply chat template - DISABLE THINKING
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # CRITICAL: Disable thinking tags
        )
        
        # Tokenize and generate
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.8,  # Slightly higher for creativity
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return response.strip()

class CompetitionCLI:
    """Simplified CLI for the competition system"""
    
    def __init__(self):
        self.question_agent = None
        self.answer_agent = None
        self.session_log = []
    
    def display_banner(self):
        """Display competition banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                  🏆 HACKATHON COMPETITION TOOL 🏆           ║
║                                                              ║
║  🎯 Question Agent: Generate championship-level MCQs        ║
║  🧠 Answer Agent: Solve with perfect accuracy               ║
║  ⚡ Token Optimized: <100 tokens per question               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(banner)
    
    def main_menu(self):
        """Display main menu and handle user choice"""
        while True:
            print("\n" + "="*60)
            print("🚀 COMPETITION MENU")
            print("="*60)
            print("1. 🎯 Question Agent - Generate MCQ")
            print("2. 🧠 Answer Agent - Solve MCQ")
            print("3. 🔄 Battle Mode - Q vs A Agent")
            print("4. 📊 Session Statistics")
            print("5. 🔧 Agent Status")
            print("6. 📝 Save Session Log")
            print("7. ❌ Exit")
            print("="*60)
            
            choice = input("🎮 Enter your choice (1-7): ").strip()
            
            if choice == "1":
                self.question_mode()
            elif choice == "2":
                self.answer_mode()
            elif choice == "3":
                self.battle_mode()
            elif choice == "4":
                self.show_statistics()
            elif choice == "5":
                self.show_agent_status()
            elif choice == "6":
                self.save_session_log()
            elif choice == "7":
                print("👋 Goodbye! Good luck in the competition!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def question_mode(self):
        """Question generation mode - simplified to Number Series only"""
        print("\n🎯 QUESTION GENERATION MODE")
        print("-" * 40)
        print("Generating championship-level Number Series question...")
        
        # Initialize question agent if needed
        if self.question_agent is None:
            self.question_agent = CompetitionAgent("question")
        
        start_time = time.time()
        
        try:
            question = self.question_agent.generate_question("championship")
            generation_time = time.time() - start_time
            
            print(f"\n✅ Question generated in {generation_time:.2f}s")
            print("="*60)
            
            if "error" in question:
                print(f"❌ Error: {question['error']}")
                print(f"Raw response: {question.get('raw_response', 'N/A')[:200]}...")
            else:
                self._display_question(question)
                
                # Log the question
                self.session_log.append({
                    "type": "question_generated",
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "generation_time": generation_time
                })
                
        except Exception as e:
            print(f"❌ Error generating question: {e}")
            import traceback
            traceback.print_exc()
    
    def answer_mode(self):
        """Answer solving mode"""
        print("\n🧠 ANSWER SOLVING MODE")
        print("-" * 40)
        
        # Initialize answer agent if needed
        if self.answer_agent is None:
            self.answer_agent = CompetitionAgent("answer")
        
        print("Choose input method:")
        print("1. Manual input")
        print("2. Load from recent question")
        
        input_choice = input("Select method (1-2): ").strip()
        
        if input_choice == "1":
            question_data = self._get_manual_question()
        elif input_choice == "2":
            question_data = self._get_recent_question()
        else:
            print("❌ Invalid choice")
            return
        
        if not question_data:
            return
        
        print(f"\n🔍 Solving question...")
        start_time = time.time()
        
        try:
            answer = self.answer_agent.solve_question(question_data)
            solving_time = time.time() - start_time
            
            print(f"\n✅ Question solved in {solving_time:.2f}s")
            print("="*60)
            
            if "error" in answer:
                print(f"❌ Error: {answer['error']}")
                print(f"Raw response: {answer.get('raw_response', 'N/A')[:200]}...")
            else:
                print(f"🎯 Answer: {answer.get('answer', 'N/A')}")
                print(f"💭 Reasoning: {answer.get('reasoning', 'N/A')}")
                
                # Check if answer is correct (if we have the expected answer)
                correct = None
                if "answer" in question_data:
                    expected = question_data["answer"].upper()
                    given = answer.get("answer", "").upper()
                    correct = expected == given
                    print(f"✅ Correctness: {'Correct' if correct else 'Incorrect'} (Expected: {expected})")
                
                # Log the answer
                self.session_log.append({
                    "type": "question_solved",
                    "timestamp": datetime.now().isoformat(),
                    "question": question_data,
                    "answer": answer,
                    "solving_time": solving_time,
                    "correct": correct
                })
                
        except Exception as e:
            print(f"❌ Error solving question: {e}")
            import traceback
            traceback.print_exc()
    
    def battle_mode(self):
        """Battle mode - Q agent vs A agent"""
        print("\n🔄 BATTLE MODE - Q AGENT vs A AGENT")
        print("-" * 50)
        
        # Initialize both agents
        if self.question_agent is None:
            self.question_agent = CompetitionAgent("question")
        if self.answer_agent is None:
            self.answer_agent = CompetitionAgent("answer")
        
        num_rounds = int(input("Enter number of battle rounds (1-10): ") or "3")
        
        results = {"correct": 0, "total": 0, "avg_time": 0, "token_violations": 0}
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n🥊 ROUND {round_num}/{num_rounds}")
            print("-" * 30)
            
            # Q-agent generates question
            print("🎯 Q-Agent generating question...")
            question = self.question_agent.generate_question("championship")
            
            if "error" in question:
                print("❌ Q-Agent failed to generate valid question")
                continue
            
            print("✅ Question generated!")
            self._display_question(question, show_answer=False)
            
            # Check token limit
            if not question.get("token_valid", True):
                print("⚠️  Token limit violation!")
                results["token_violations"] += 1
            
            # A-agent solves question  
            print("\n🧠 A-Agent solving...")
            start_time = time.time()
            answer = self.answer_agent.solve_question(question)
            solve_time = time.time() - start_time
            
            if "error" in answer:
                print("❌ A-Agent failed to solve")
                continue
            
            # Check correctness
            expected = question.get("answer", "").upper()
            given = answer.get("answer", "").upper()
            correct = expected == given
            
            print(f"🎯 A-Agent answered: {given}")
            print(f"✅ Expected: {expected}")
            print(f"⏱️  Time: {solve_time:.2f}s")
            print(f"🏆 Result: {'CORRECT' if correct else 'INCORRECT'}")
            
            # Update results
            results["total"] += 1
            if correct:
                results["correct"] += 1
            results["avg_time"] += solve_time
            
            # Log battle round
            self.session_log.append({
                "type": "battle_round",
                "timestamp": datetime.now().isoformat(),
                "round": round_num,
                "question": question,
                "answer": answer,
                "correct": correct,
                "solve_time": solve_time
            })
        
        # Display battle results
        if results["total"] > 0:
            accuracy = (results["correct"] / results["total"]) * 100
            avg_time = results["avg_time"] / results["total"]
            
            print(f"\n🏆 BATTLE RESULTS")
            print("="*40)
            print(f"Accuracy: {results['correct']}/{results['total']} ({accuracy:.1f}%)")
            print(f"Average Time: {avg_time:.2f}s")
            print(f"Token Violations: {results['token_violations']}/{results['total']}")
            print(f"Performance: {'🔥 EXCELLENT' if accuracy >= 70 else '⚡ GOOD' if accuracy >= 50 else '💪 NEEDS WORK'}")
    
    def show_statistics(self):
        """Show session statistics"""
        print("\n📊 SESSION STATISTICS")
        print("="*50)
        
        if not self.session_log:
            print("No activity in this session yet.")
            return
        
        questions_generated = len([x for x in self.session_log if x["type"] == "question_generated"])
        questions_solved = len([x for x in self.session_log if x["type"] == "question_solved"])
        battle_rounds = len([x for x in self.session_log if x["type"] == "battle_round"])
        
        print(f"Questions Generated: {questions_generated}")
        print(f"Questions Solved: {questions_solved}")
        print(f"Battle Rounds: {battle_rounds}")
        
        # Calculate accuracy if we have solved questions
        solved_entries = [x for x in self.session_log if x["type"] in ["question_solved", "battle_round"] and x.get("correct") is not None]
        if solved_entries:
            correct_answers = len([x for x in solved_entries if x["correct"]])
            accuracy = (correct_answers / len(solved_entries)) * 100
            print(f"Answer Accuracy: {correct_answers}/{len(solved_entries)} ({accuracy:.1f}%)")
        
        # Show recent activity
        print(f"\nRecent Activity:")
        for log_entry in self.session_log[-5:]:
            timestamp = log_entry["timestamp"].split("T")[1][:8]
            print(f"  {timestamp} - {log_entry['type'].replace('_', ' ').title()}")
    
    def show_agent_status(self):
        """Show agent loading status"""
        print("\n🔧 AGENT STATUS")
        print("="*40)
        
        q_status = "✅ Loaded" if self.question_agent and self.question_agent.model else "❌ Not Loaded"
        a_status = "✅ Loaded" if self.answer_agent and self.answer_agent.model else "❌ Not Loaded"
        
        print(f"Question Agent: {q_status}")
        print(f"Answer Agent: {a_status}")
        
        if self.question_agent:
            q_checkpoint = "✅ Found" if os.path.exists(self.question_agent.checkpoint_dir) else "❌ Missing"
            print(f"Q-Agent Checkpoint: {q_checkpoint} ({self.question_agent.checkpoint_dir})")
            
        if self.answer_agent:
            a_checkpoint = "✅ Found" if os.path.exists(self.answer_agent.checkpoint_dir) else "❌ Missing"
            print(f"A-Agent Checkpoint: {a_checkpoint} ({self.answer_agent.checkpoint_dir})")
    
    def save_session_log(self):
        """Save session log to file"""
        if not self.session_log:
            print("No activity to save.")
            return
        
        filename = f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.session_log, f, indent=2)
        
        print(f"✅ Session log saved to {filename}")
    
    def _display_question(self, question: Dict, show_answer: bool = True):
        """Display a formatted question"""
        print(f"📝 Topic: {question.get('topic', 'N/A')}")
        print(f"❓ Question: {question.get('question', 'N/A')}")
        
        choices = question.get('choices', [])
        for choice in choices:
            print(f"   {choice}")
        
        if show_answer:
            print(f"✅ Answer: {question.get('answer', 'N/A')}")
            print(f"💡 Explanation: {question.get('explanation', 'N/A')}")
        
        if question.get('token_count'):
            status = "✅" if question.get('token_valid') else "❌"
            print(f"🔢 Tokens: {question['token_count']}/100 {status}")
    
    def _get_manual_question(self) -> Dict:
        """Get question details from manual input"""
        print("\n📝 Enter question details:")
        
        topic = input("Topic [Number Series]: ").strip() or "Number Series"
        question_text = input("Question: ").strip()
        
        choices = []
        for i, letter in enumerate(['A', 'B', 'C', 'D']):
            choice = input(f"Choice {letter}: ").strip()
            if not choice.startswith(f"{letter})"):
                choice = f"{letter}) {choice}"
            choices.append(choice)
        
        answer = input("Correct answer (A/B/C/D): ").strip().upper()
        
        return {
            "topic": topic,
            "question": question_text,
            "choices": choices,
            "answer": answer
        }
    
    def _get_recent_question(self) -> Optional[Dict]:
        """Get a recently generated question"""
        recent_questions = [x for x in self.session_log if x["type"] == "question_generated"]
        
        if not recent_questions:
            print("❌ No recent questions found. Generate a question first.")
            return None
        
        print("\nRecent questions:")
        for i, log_entry in enumerate(recent_questions[-5:], 1):
            q = log_entry["question"]
            print(f"{i}. {q.get('question', 'N/A')[:50]}...")
        
        try:
            choice = int(input("Select question (1-5): ")) - 1
            if 0 <= choice < min(5, len(recent_questions)):
                return recent_questions[-(5-choice)]["question"]
            else:
                print("❌ Invalid selection")
                return None
        except ValueError:
            print("❌ Invalid input")
            return None
    
    def run(self):
        """Main entry point"""
        self.display_banner()
        self.main_menu()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="🏆 Hackathon Competition CLI Tool")
    parser.add_argument("--test", action="store_true", help="Run quick system test")
    
    args = parser.parse_args()
    
    if args.test:
        run_system_test()
    else:
        # Run interactive CLI
        cli = CompetitionCLI()
        cli.run()

def run_system_test():
    """Run comprehensive system test"""
    print("🧪 Running System Test...")
    print("="*50)
    
    try:
        # Test Question Agent
        print("1. Testing Question Agent...")
        q_agent = CompetitionAgent("question")
        question = q_agent.generate_question("championship")
        
        if "error" in question:
            print(f"❌ Question Agent Failed: {question['error']}")
            return
        else:
            print("✅ Question Agent Working")
            print(f"   Generated: {question.get('question', 'N/A')}...")
            print(f"   Token Count: {question.get('token_count', 'N/A')}/100")
        
        # Test Answer Agent
        print("\n2. Testing Answer Agent...")
        a_agent = CompetitionAgent("answer")
        answer = a_agent.solve_question(question)
        
        if "error" in answer:
            print(f"❌ Answer Agent Failed: {answer['error']}")
            return
        else:
            print("✅ Answer Agent Working")
            print(f"   Answer: {answer.get('answer', 'N/A')}")
            print(f"   Reasoning: {answer.get('reasoning', 'N/A')}...")
        
            # Check correctness
            expected = question.get("answer", "").upper()
            given = answer.get("answer", "").upper()
            correct = expected == given
            
            print(f"\n3. System Integration Test:")
            print(f"   Expected: {expected}")
            print(f"   Generated: {given}")
            print(f"   Result: {'✅ PASS' if correct else '❌ FAIL'}")
            
            # Performance summary
            print(f"\n🏆 System Status: {'🔥 READY FOR COMPETITION' if correct else '⚠️ NEEDS ATTENTION'}")
        
    except Exception as e:
        print(f"❌ System Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()  # Properly indented call to main()