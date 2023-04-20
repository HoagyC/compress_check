import copy
import json
import os
import time
from typing import List, Tuple, Dict

import numpy as np

import openai

MODEL_CONDITION_GENERATION = "gpt-3.5-turbo"
openai.api_key = json.load(open("secrets.json"))["openai_key"]

def query_openai(message: str, model: str = MODEL_CONDITION_GENERATION, max_tokens: int = 100):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages = message,
                max_tokens=40,
            )
            return response
        except (openai.error.InvalidRequestError, openai.error.RateLimitError):
            print("OpenAI error, retrying...")
            time.sleep(1)
            continue


question_examples: List[Tuple[str, str, str]] = [
    ("An appeal court judge in Moscow has ruled to keep Evan Gershkovich, the Wall Street Journal reporter arrested on espionage charges, in pre-trial detention for at least two months.",
     "Was the reporter arrested on fraud charges?",
     "No"),
     ("The nation's second-largest bank by assets said profit in the three months to March rose 15 per cent to $8.2bn, or $0.94 a share, from the same period a year ago. Analysts had predicted profit would slip in the first quarter.",
      "Did the bank report a profit of more than $0.90 per share?",
      "Yes"),
]

def make_question_message(examples: List[Tuple[str, str, str]] = []) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please create difficult binary questions about a given text which are capable of being answered with a clear \"Yes\" or \"No\"."},
    ]
    for text, question, answer in examples:
        messages.append({"role": "user", "content": f"Text: '{text}', Answer: '{answer}', Question: '{question}'"})
    return messages

def make_questions(verbose: bool) -> None:
    examples: List[Dict[str, any]] = []

    with open("paragraphs.txt", "r") as f:
        paragraphs = f.readlines()
    
    for paragraph in paragraphs:
        examples += [{"text": paragraph} for _ in range(10)]
    
    print(f"Got {len(examples)} total examples from {len(paragraphs)} paragraphs.")

    # Use OpenAI to generate binary questions about the paragraphs
    base_message = make_question_message(question_examples)
    for example_dict in examples:
        while True:
            message = copy.deepcopy(base_message)
            answer = "Yes" if np.random.random() > 0.5 else "No" # Randomly choose the answer because otherwise it always gives true questions
            message.append({"role": "user", "content": f"Text: '{example_dict['text']}', Answer: '{answer}', Question:"})

            response = query_openai(message)

            if "?" not in response.choices[0]["message"]["content"]:
                continue

            question = response.choices[0]["message"]["content"].split("?")[0] + "?"
            example_dict["question"] = question
            example_dict["answer"] = answer

            if verbose:
                print(f"Got question {question} with answer {example_dict['answer']}")
            
            break
            

    # Use GPT-4 to compress a paragraph into a single sentence
    prompt = "'Please aggressively compress the following paragraph as much as you can, below 30 characters if possible, removing spaces, making use of emoji and shorthand wherever possible while trying to make it legible to you (GPT-4). Don't worry about making it interpretible to a human."
    for example_dict in examples:
        compress_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"Text: '{example_dict['text']}'"},
        ]
        response = query_openai(compress_messages)

        example_dict["compressed"] = response.choices[0]["message"]["content"]
        if verbose:
            print(f"Got compressed paragraph {example_dict['compressed']}")

    # Use GPT-4 to try to uncompress the compressed paragraph
    prompt_lines = ["The following text is a compressed version of a paragraph that was compressed by asking GPT-4 the following command: ",
                    "'Please aggressively compress the following paragraph as much as you can, removing spaces, making use of emoji and shorthand wherever possible while ensuring it's still legible to you (GPT-4). ",
                    "Don't worry about making it interpretible to a human.' Please uncompress this back into a legible paragraph."
    ]
    prompt = "".join(prompt_lines)
    for example_dict in examples:
        uncompress_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"Text: '{example_dict['compressed']}'"},
        ]
        response = query_openai(uncompress_messages)
    
        example_dict["uncompressed"] = response.choices[0]["message"]["content"]
        if verbose:
            print(f"Got uncompressed paragraph {example_dict['uncompressed']}")

    # Save progress
    with open("examples.json", "w") as f:
        json.dump(examples, f)
    
    return examples

def answer_questions(examples: List[Dict[str, any]], verbose: bool) -> None:
    # Use GPT-4 to try to answer the questions using the compressed paragraph
    for example_dict in examples:
        prompt_lines = [
            f"The following text is a compressed version of a paragraph that was compressed by asking GPT-4 to compress a paragraph down to 100 characters:",
            example_dict['compressed'],
            f"Please answer the following question using the compressed paragraph:",
            example_dict['question'],
            "Please answer with only 'Yes' or 'No'. If it is totally impossible to know, please answer 'DK'."
            ]
        prompt = "\n".join(prompt_lines)
        answer_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        while True:
            response = query_openai(answer_messages)

            compressed_answer = response.choices[0]["message"]["content"]
            if "Yes" in compressed_answer:
                example_dict["compressed_answer"] = "Yes"
            elif "No" in compressed_answer:
                example_dict["compressed_answer"] = "No"
            else:
                continue
            if verbose:
                print(f"Got answer {compressed_answer} for question {example_dict['question']}")
            break

    # Use GPT-4 to try to answer the questions using the uncompressed paragraph
    for example_dict in examples:
        prompt_lines = [
            f"The following text is a paragraph that was compressed and then uncompressed by asking GPT-4:",
            example_dict['uncompressed'],
            f"Please answer the following question using the compressed paragraph:",
            example_dict['question'],
            "Please answer with only 'Yes' or 'No', giving your best guess if it doesn't seem possible. If it is totally impossible to know, please answer 'DK'."
            ]
        prompt = "\n".join(prompt_lines)
        answer_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        while True:
            response = query_openai(answer_messages)

            uncompressed_answer = response.choices[0]["message"]["content"]
            if "Yes" in uncompressed_answer:
                example_dict["uncompressed_answer"] = "Yes"
            elif "No" in uncompressed_answer:
                example_dict["uncompressed_answer"] = "No"
            else:
                print(f"Got answer {uncompressed_answer}")
                continue
            if verbose:
                print(f"Got answer {uncompressed_answer} for question {example_dict['question']}")
            break
    
    # Score the results
    for example_dict in examples:
        example_dict["compressed_score"] = 0
        example_dict["uncompressed_score"] = 0
        if example_dict["compressed_answer"] == example_dict["answer"]:
            example_dict["compressed_score"] = 1
        if example_dict["uncompressed_answer"] == example_dict["answer"]:
            example_dict["uncompressed_score"] = 1
        
    

    # Print the results
    print("Results:")
    compressed_average = np.mean([example_dict["compressed_score"] for example_dict in examples])
    uncompressed_average = np.mean([example_dict["uncompressed_score"] for example_dict in examples])
    print(f"Compressed average: {compressed_average}")
    print(f"Uncompressed average: {uncompressed_average}")
    
    # Save progress
    with open("examples_answered.json", "w") as f:
        json.dump(examples, f)



def main(verbose: bool = False):
    if os.path.exists("examples.json"):
        with open("examples.json", "r") as f:
            examples = json.load(f)
    else:
        exmamples = make_questions(verbose)
    
    answer_questions(examples, verbose)
    

if __name__ == "__main__":
    main(verbose=True)