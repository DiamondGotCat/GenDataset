import json
import random
from ollama import chat
from rich import print
from rich.prompt import Prompt

def autoPadding(input):
    splitted = input.split("\n")
    output = ""

    for line in splitted:
        output += "        " + line + "\n"
    
    return output

def swap_roles(chat_history):
    swapped_history = []
    for turn in chat_history:
        new_role = "assistant" if turn["role"] == "user" else "user"
        swapped_turn = {
            "role": new_role,
            "content": turn["content"]
        }
        swapped_history.append(swapped_turn)
    return swapped_history

def getOneResponse(chat_history, model_id):
    response = chat(model=model_id, messages=chat_history)
    return response['message']['content']

def generateFirstQuestion(model_id):
    result = getOneResponse([{"role": "user", "content": """
Just create one question.
I don't need your answer, just the question.
                              
"""}], model_id)
    return result

def generateQuestion(chat_history, model_id):
    result = getOneResponse([{"role": "user", "content": f"""
Just create one next question.
I don't need your answer, just the question.
(Please output question, with Plain Text. Not JSON.)

History:
{json.dumps(chat_history).replace("user", "question").replace("assistant", "answer")}

{random.choices(range(1,999999))}
"""}], model_id)
    return result

def generateAnswer(chat_history, model_id):
    result = getOneResponse(chat_history, model_id)
    return result

def generateOneConversation(model_id):
    global assistant_turns_for_one_conversation
    conversation = []
    first = generateFirstQuestion(model_id)
    conversation.append({"role": "user", "content": first})
    print(f"    [blue]USER (1/{(assistant_turns_for_one_conversation * 2) - 2})[/blue]")
    print(autoPadding(first))
    isCurrentUserTurn = False
    for i in range(1, (assistant_turns_for_one_conversation * 2) - 2): # Do not contain Fist Question for range (with " - 2")
        if isCurrentUserTurn:
            content = generateQuestion(conversation, model_id)
            conversation.append({"role": "user", "content": content})
            print(f"    [blue]USER ({i+1}/{(assistant_turns_for_one_conversation * 2) - 2})[/blue]")
            print(autoPadding(content))
            isCurrentUserTurn = False
        else:
            content = generateAnswer(conversation, model_id)
            conversation.append({"role": "assistant", "content": content})
            print(f"    [blue]ASSISTANT ({i+1}/{(assistant_turns_for_one_conversation * 2) - 2})[/blue]")
            print(autoPadding(content))
            isCurrentUserTurn = True

    return conversation

# Please use this Function for Process-only Usage
def main_process(model_id):
    global dataset
    global conversations_for_one_dataset
    global assistant_turns_for_one_conversation
    
    dataset = [] # [ dataset: [ conversation: { turn: "role", "content" }, ... ], ... ]
    conversations_for_one_dataset = 5 # 1,2,3,...
    assistant_turns_for_one_conversation = 2 # Set 2 to One Set Q&A Only.

    print("[blue]GenDataset has Started![/blue]")
    print(f"Conversations for One Dataset: {conversations_for_one_dataset}")
    print(f"Assistant Turns for One Conversations: {assistant_turns_for_one_conversation}")
    print()

    for i in range(1, conversations_for_one_dataset):
        print(f"[green]CONVERSATION ({i}/{conversations_for_one_dataset})[/green]")
        dataset.append(generateOneConversation(model_id))

    return dataset

def main():

    model_id = Prompt.ask("Model ID on Ollama")
    export_path = Prompt.ask("File Path(Export)")

    # Main Process
    result = main_process(model_id)

    # Export
    with open(export_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
