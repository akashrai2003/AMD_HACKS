import os
import torch
import argparse
import re
import json
import time
from typing import Optional, List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback
)
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
import wandb

class MCQReasoningTrainer:
    """Trainer class for structured MCQ reasoning with JSON output format."""
    
    def __init__(self, args):
        """Initialize the trainer with configuration arguments."""
        self.args = args
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        
        # Inference model cache
        self._inference_model = None
        self._inference_tokenizer = None
        
        self.system_prompt = """You are an MCQ answering model.
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
}

"""
        # Setup environment
        self._setup_environment()
        
        # Display configuration
        self._display_config()
    
    def _setup_environment(self):
        """Setup environment variables and GPU configuration."""
        # GPU selection - parse gpu_ids from args
        gpu_ids = [int(x.strip()) for x in self.args.gpu_ids.split(',') if x.strip().isdigit()]
        if not gpu_ids:
            gpu_ids = [0]  # Default to GPU 0 if no valid IDs provided
        
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpu_ids)))
        
        # Add environment variables for distributed training
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
    
    def _display_config(self):
        """Display training configuration."""
        print("\n" + "="*60)
        print("MCQ STRUCTURED REASONING TRAINER")
        print("="*60)
        print(f"Model: {self.args.model_name}")
        print(f"Output Directory: {self.args.output_dir}")
        print(f"Dataset: {self.args.dataset_file}")
        print(f"Training Mode: SFT")
        print(f"LoRA Config: r={self.args.lora_r}, alpha={self.args.lora_alpha}")
        print(f"Learning Rate: {self.args.learning_rate}")
        print(f"Epochs: {self.args.num_train_epochs}")
        print(f"Batch Size: {self.args.per_device_train_batch_size}")
        print("="*60 + "\n")

    def load_dataset(self):
        """Load and process the custom dataset."""
        print(f"Loading dataset from: {self.args.dataset_file}")
        items = []

        try:
            with open(self.args.dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print("Error: Dataset should be a JSON list")
                raise ValueError("Invalid dataset format")
            
            for idx, item in enumerate(data):
                try:
                    # Validate required keys
                    required_keys = ["question", "options", "question_type", "intermediate_steps", "answer"]
                    if not all(key in item for key in required_keys):
                        print(f"Warning: Missing keys in item {idx}. Required: {required_keys}")
                        continue
                    
                    # Extract data
                    question_text = item["question"]
                    options = item["options"]
                    question_type = item["question_type"]
                    intermediate_steps = item["intermediate_steps"]
                    answer = item["answer"]
                    
                    # Format question with options
                    if isinstance(options, dict):
                        choices_text = []
                        for choice_key in sorted(options.keys()):
                            choices_text.append(f"{choice_key}) {options[choice_key]}")
                        formatted_question = f"Question: {question_text}\nChoices:\n" + "\n".join(choices_text)
                    elif isinstance(options, list):
                        formatted_question = f"Question: {question_text}\nChoices:\n" + "\n".join(options)
                    else:
                        print(f"Warning: Invalid options format in item {idx}")
                        continue
                    
                    items.append({
                        'formatted_question': formatted_question,
                        'question_type': question_type,
                        'intermediate_steps': intermediate_steps,
                        'answer': answer,
                        'original_question': question_text,
                        'original_options': options
                    })
                    
                    if idx <= 2:  # Show first 3 for debugging
                        print(f"✓ Loaded item {idx}: {question_text[:50]}...")
                        
                except Exception as e:
                    print(f"Warning: Error processing item at index {idx}: {e}")
                    continue

        except FileNotFoundError:
            print(f"Error: Dataset file {self.args.dataset_file} not found.")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {self.args.dataset_file}: {e}")
            raise
        except Exception as e:
            print(f"Error: Unexpected error loading dataset: {e}")
            raise
    
        if not items:
            print("Error: No valid items found in dataset file.")
            raise ValueError("No valid training data found")
        
        print(f"Successfully loaded {len(items)} items from JSON file")
        self.raw_items = items
        return items

    def _format_sft_dataset(self, raw_items: List[Dict[str, str]]) -> Dataset:
        """Format raw items for SFT training with structured JSON output."""
        # Load tokenizer for formatting
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        formatted_items = []
        
        for item in raw_items:
            formatted_question = item['formatted_question']
            question_type = item['question_type']
            intermediate_steps = item['intermediate_steps']
            answer = item['answer']
            
            # Create the expected JSON output structure
            expected_output = {
                "question_type": question_type,
                "intermediate_steps": intermediate_steps,
                "answer": answer
            }
            
            # Convert to formatted JSON string
            json_output = json.dumps(expected_output, indent=2, ensure_ascii=False)
            
            # Format as chat
            chat_messages = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': formatted_question},
                {'role': 'assistant', 'content': json_output}
            ]
            
            # Apply chat template
            formatted_text = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            formatted_items.append({
                "text": formatted_text,
                "question": formatted_question,
                "answer": json_output
            })
        
        print(f"Created {len(formatted_items)} SFT training samples")
        return Dataset.from_list(formatted_items)
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer."""
        print(f"Loading model: {self.args.model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        )

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
                
        # Setup model configuration
        self.model.config.use_cache = False
        if hasattr(self.model.config, 'pretraining_tp'):
            self.model.config.pretraining_tp = 1
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name, 
            trust_remote_code=True, 
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = "right"
        self.tokenizer.model_max_length = self.args.max_seq_length
        
        print("Model and tokenizer loaded successfully")
    
    def setup_peft_config(self) -> LoraConfig:
        """Setup LoRA configuration."""
        return LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )
    
    def setup_wandb(self):
        """Setup Weights & Biases logging."""
        if not self.args.disable_wandb:
            try:
                wandb.init(
                    project=self.args.wandb_project,
                    name=self.args.wandb_run_name,
                    config={
                        "model_name": self.args.model_name,
                        "learning_rate": self.args.learning_rate,
                        "batch_size": self.args.per_device_train_batch_size,
                        "epochs": self.args.num_train_epochs,
                        "lora_r": self.args.lora_r,
                        "lora_alpha": self.args.lora_alpha,
                        "training_type": "sft"
                    }
                )
                print("Weights & Biases initialized successfully")
            except Exception as e:
                print(f"WandB initialization failed: {e}. Training will continue without WandB.")
                self.args.disable_wandb = True

    def train_sft(self):
        """Train using Supervised Fine-Tuning."""
        print("Starting SFT training...")
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            save_strategy="steps",
            save_steps=100,
            logging_steps=10,
            learning_rate=self.args.learning_rate,
            weight_decay=0.01,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="wandb" if not self.args.disable_wandb else "none",
        )
        
        # Setup LoRA config
        peft_config = self.setup_peft_config()
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=training_args,
            train_dataset=self.dataset,
            peft_config=peft_config,
        )
        
        # Start training
        self.trainer.train()
        
        # Save model
        print(f"Saving LoRA adapters to {self.args.output_dir}")
        self.trainer.save_model(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        
        print("SFT training completed successfully")

    def train(self):
        """Main training function."""
        print("Starting SFT training...")
        
        # Load dataset
        raw_items = self.load_dataset()
        
        # Format dataset
        self.dataset = self._format_sft_dataset(raw_items)
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Setup wandb
        self.setup_wandb()
        
        # Train
        self.train_sft()
        
        # Cleanup
        if not self.args.disable_wandb and wandb.run:
            wandb.finish()
        
        print("SFT training completed!")

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the output directory."""
        if not os.path.exists(self.args.output_dir):
            print(f"Output directory {self.args.output_dir} does not exist")
            return None
        
        checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
        checkpoints = []
        
        for item in os.listdir(self.args.output_dir):
            item_path = os.path.join(self.args.output_dir, item)
            if os.path.isdir(item_path):
                match = checkpoint_pattern.match(item)
                if match:
                    step_num = int(match.group(1))
                    checkpoints.append((step_num, item_path))
        
        if not checkpoints:
            print(f"No checkpoints found in {self.args.output_dir}")
            return None
        
        checkpoints.sort(key=lambda x: x[0])
        latest_checkpoint = checkpoints[-1][1]
        
        print(f"Found {len(checkpoints)} checkpoints. Using latest: {latest_checkpoint}")
        return latest_checkpoint

    def extract_json_answer(self, text: str) -> Dict[str, Any]:
        """Extract structured JSON answer from model output."""
        try:
            # Try to find JSON content in the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed
            else:
                print(f"No JSON found in: {text[:100]}...")
                return {"question_type": "", "intermediate_steps": {}, "answer": ""}
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {"question_type": "", "intermediate_steps": {}, "answer": ""}

    def setup_inference_model(self):
        """Setup model once for inference mode."""
        if hasattr(self, '_inference_model') and self._inference_model is not None:
            return  # Already loaded
        
        try:
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint:
                print(f"Loading model from checkpoint: {latest_checkpoint}")
                
                # Load base model
                self._inference_model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # Load LoRA weights
                self._inference_model = PeftModel.from_pretrained(
                    self._inference_model,
                    latest_checkpoint,
                    torch_dtype=torch.bfloat16,
                )
                
                print("✓ LoRA weights loaded successfully")
            else:
                print("No checkpoint found, loading base model...")
                self._inference_model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            # Load tokenizer
            self._inference_tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name, 
                trust_remote_code=True
            )
            
            if self._inference_tokenizer.pad_token is None:
                self._inference_tokenizer.pad_token = self._inference_tokenizer.eos_token
            
            print("Inference model setup completed")
            
        except Exception as e:
            print(f"Error setting up inference model: {e}")
            raise

    def test_single_inference(self):
        """Test inference on a single question."""
        if not self.args.test_question:
            print("Error: No test question provided")
            return
        
        self.setup_inference_model()
        
        print("\n" + "="*60)
        print("🔍 SINGLE QUESTION INFERENCE TEST")
        print("="*60)
        
        # Format the test question
        test_input = self.args.test_question
        
        # Create chat messages
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': test_input}
        ]
        
        # Apply chat template
        formatted_input = self._inference_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self._inference_tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Generate
        with torch.no_grad():
            outputs = self._inference_model.generate(
                inputs.input_ids.to(self._inference_model.device),
                attention_mask=inputs.attention_mask.to(self._inference_model.device),
                max_new_tokens=512,
                temperature=self.args.temperature,
                do_sample=True,
                pad_token_id=self._inference_tokenizer.eos_token_id
            )
        
        # Decode response
        response = self._inference_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"**Question:** {test_input}")
        print(f"\n**Raw Model Response:**\n{response}")
        
        # Extract structured answer
        structured_answer = self.extract_json_answer(response)
        print(f"\n**Extracted Answer:** {structured_answer.get('answer', 'N/A')}")
        print(f"**Question Type:** {structured_answer.get('question_type', 'N/A')}")
        print(f"**Intermediate Steps:** {json.dumps(structured_answer.get('intermediate_steps', {}), indent=2)}")
        
        # Save results if output file specified
        if self.args.inference_output:
            with open(self.args.inference_output, 'w') as f:
                f.write(f"# MCQ Structured Reasoning Inference Results\n\n")
                f.write(f"Model: {self.args.model_name}\n")
                f.write(f"Checkpoint: {self.find_latest_checkpoint()}\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Question:** {test_input}\n\n")
                f.write(f"**Model Response:**\n```json\n{response}\n```\n\n")
                f.write(f"**Extracted Answer:** {structured_answer.get('answer', 'N/A')}\n")
                f.write(f"**Question Type:** {structured_answer.get('question_type', 'N/A')}\n")
                f.write(f"**Intermediate Steps:**\n```json\n{json.dumps(structured_answer.get('intermediate_steps', {}), indent=2)}\n```\n")
            
            print(f"Results saved to: {self.args.inference_output}")

    def cleanup_inference_model(self):
        """Clean up inference model to free memory."""
        if hasattr(self, '_inference_model') and self._inference_model is not None:
            del self._inference_model
            self._inference_model = None
        if hasattr(self, '_inference_tokenizer') and self._inference_tokenizer is not None:
            del self._inference_tokenizer
            self._inference_tokenizer = None
        torch.cuda.empty_cache()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MCQ Structured Reasoning Trainer")
    
    # General arguments
    parser.add_argument("--model_name", type=str, default="/jupyter-tutorial/hf_models/Qwen3-4B", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="ckpt/sft", help="Output directory for checkpoints and model")
    parser.add_argument("--dataset_file", type=str, default="full_dataset.json", help="Path to the dataset file (JSON)")
    parser.add_argument("--test_question", type=str, help="Single question for testing inference")
    parser.add_argument("--inference_output", type=str, default="inference.md", help="Output file for inference results")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs to use (comma-separated)")
    
    # Training arguments
    parser.add_argument("--training_type", type=str, default="sft", help="Type of training (always SFT)")
    parser.add_argument("--mode", type=str, choices=["train", "inference", "both"], default="train", help="Mode of operation")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for response generation")
    
    # WandB arguments
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="mcq_reasoning", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, help="WandB run name")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Auto-generate wandb run name if not provided
    model_display_name = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{model_display_name}-sft-r{args.lora_r}-lr{args.learning_rate}"
    
    # Create trainer
    trainer = MCQReasoningTrainer(args)
    
    try:
        # Run based on mode
        if args.mode == 'train':
            trainer.train()
        
        elif args.mode == 'inference':
            if args.test_question:
                trainer.test_single_inference()
            else:
                print("Error: No test question provided for inference mode")
        
        elif args.mode == 'both':
            # Train first
            trainer.train()
            print("\n" + "="*60)
            print("TRAINING COMPLETED - STARTING INFERENCE")
            print("="*60)
            
            # Run inference if test question provided
            if args.test_question:
                trainer.test_single_inference()
            else:
                print("No test question provided, skipping inference")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if not args.disable_wandb and wandb.run:
            wandb.finish()
        trainer.cleanup_inference_model()

if __name__ == "__main__":
    """
    MCQ STRUCTURED REASONING TRAINER
    ================================

    This script provides SFT training for MCQ models with structured JSON reasoning output.
    The model learns to generate question_type, intermediate_steps, and answer in JSON format.

    USAGE EXAMPLE:
    
    python -m trainer \
        --training_type sft \
        --mode both \
        --model_name /jupyter-tutorial/hf_models/Qwen3-4B \
        --output_dir ckpt/sft \
        --inference_output \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --lora_r 32 \
        --lora_alpha 64 \
        --dataset_file full_dataset.json \
        --test_question "Your test question here"

         DATASET FORMAT:
     The dataset should be a JSON list where each item contains:
     - "question": Question text
     - "options": Dict or list of answer choices  
     - "question_type": One of "blood_relations"/"seating_arrangement"/"truth_lie"
     - "intermediate_steps": JSON object with reasoning steps
     - "answer": Correct answer letter ("A", "B", "C", or "D")

    OUTPUT FORMAT:
    The model generates structured JSON with:
    - "question_type": Classification of question type
    - "intermediate_steps": Reasoning steps based on question type
    - "answer": Final answer choice
    """
    main()