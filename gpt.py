#!/usr/bin/env python

# Copyright (C) 2022 Andreas Stuhlmueller
# License: MIT
# SPDX-License-Identifier: MIT

import sys
import os
import argparse
import re
from pathlib import Path
from typing import Union, Optional, List
import json

APIType = Union["openai", "anthropic", "writerai"]

openai = None
anthropic = None
writerai = None

try:
    import openai
except ImportError:
    pass

try:
    import anthropic
except ImportError:
    pass

try:
    import writerai
except ImportError:
    pass

try:
    import jsonlines
except ImportError:
    jsonlines = None

def none_or_str(value):
    if value == 'None':
        return None
    return value

def none_or_json(value):
    if value == 'None':
        return None
    return json.loads(value)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("api_key", help="The API key to use for the selected API.")
    parser.add_argument(
        "model", help="The model to use (e.g., 'gpt-4', 'claude-3-sonnet-20240229')."
    )
    parser.add_argument("max_tokens", help="Max tokens value to be used with the API.")
    parser.add_argument(
        "temperature", help="Temperature value to be used with the API."
    )
    parser.add_argument(
        "api_type",
        type=str,
        choices=("openai", "anthropic", "writerai"),
        help="The type of API to use: 'openai' or 'anthropic' or 'writerai.",
    )
    parser.add_argument("prompt_file", help="The file that contains the prompt.")
    parser.add_argument("system_prompt", help="The system prompt.")
    parser.add_argument(
        "graphs_description",
        type=none_or_str,
        nargs="?",
        const=None,
        help="The description for the graphs.",
    )
    parser.add_argument("graph_ids", nargs="?", const=None, help="Graph ids.", type=none_or_json)
    parser.add_argument("image_ids", nargs="?", const=None, help="Image ids.", type=none_or_json)
    parser.add_argument(
        "application_id",
        nargs="?",
        const=None,
        type=none_or_str,
        help="The 'writerai application identifier.",
    )
    parser.add_argument("inputs", nargs="?", const=None, help="Json input for application.", type=none_or_json)
    parser.add_argument("response_schema", nargs="?", const=None, help="Json for response schema.", type=none_or_json)
    return parser.parse_args()


def read_input_text() -> str:
    """Read input text from stdin."""
    return sys.stdin.read()


def stream_openai_chat_completions(
    prompt: str,
    api_key: str,
    model: str,
    max_tokens: str,
    temperature: str,
    system_prompt: str,
) -> openai.Stream:
    """Stream chat completions from the OpenAI API."""
    if openai is None:
        print("Error: OpenAI Python package is not installed.")
        print("Please install by running `pip install openai'.")
        sys.exit(1)

    if api_key == "NOT SET":
        print("Error: OpenAI API key not set.")
        print(
            'Add (setq gpt-openai-key "sk-Aes.....AV8qzL") to your Emacs init.el file.'
        )
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)

    messages = [{"role": "system", "content": system_prompt}]
    pattern = re.compile(
        r"^(User|Assistant):(.+?)(?=\n(?:User|Assistant):|\Z)", re.MULTILINE | re.DOTALL
    )
    matches = pattern.finditer(prompt)
    for match in matches:
        role = match.group(1).lower()
        content = match.group(2).strip()
        messages.append({"role": role, "content": content})

    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            stream=True,
        )
    except openai.APIError as error:
        print(f"Error: {error}")
        sys.exit(1)


def stream_anthropic_chat_completions(
    prompt: str, api_key: str, model: str, max_tokens: str, temperature: str
) -> anthropic.Anthropic:
    """Stream chat completions from the Anthropic API."""
    if anthropic is None:
        print("Error: Anthropic Python package is not installed.")
        print("Please install by running `pip install anthropic'.")
        sys.exit(1)

    if api_key == "NOT SET":
        print("Error: Anthropic API key not set.")
        print(
            'Add (setq gpt-anthropic-key "sk-ant-api03-...") to your Emacs init.el file.'
        )
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    messages = []
    pattern = re.compile(
        r"^(User|Assistant):(.+?)(?=\n(?:User|Assistant):|\Z)", re.MULTILINE | re.DOTALL
    )
    matches = pattern.finditer(prompt)

    # Anthropic requires alternating user and assistant messages,
    # so we group user messages together
    current_user_message = None

    for match in matches:
        role = "user" if match.group(1).lower() == "user" else "assistant"
        content = match.group(2).strip()

        if role == "user":
            if current_user_message is None:
                current_user_message = content
            else:
                current_user_message += "\n\n" + content
        else:
            if current_user_message is not None:
                messages.append({"role": "user", "content": current_user_message})
                current_user_message = None
            messages.append({"role": "assistant", "content": content})

    if current_user_message is not None:
        messages.append({"role": "user", "content": current_user_message})

    try:
        return client.messages.create(
            model=model,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            stream=True,
        )
    except anthropic.APIError as error:
        print(f"Error: {error}")

        sys.exit(1)

def stream_writerai_app_completions(
    application_id: str,
    inputs: dict,
    api_key: str,
) -> writerai.Stream:
    """Stream chat completions from the Writerai API."""
    if writerai is None:
        print("Error: Writerai Python package is not installed.")
        print("Please install by running `pip install writerai'.")
        sys.exit(1)

    if api_key == "NOT SET":
        print("Error: Writerai API key not set.")
        print(
            'Add (setq gpt-writerai-key "sk-Aes.....AV8qzL") to your Emacs init.el file.'
        )
        sys.exit(1)

    client = writerai.Writer(api_key=api_key)

    try:
        return client.applications.generate_content(
            application_id=application_id,
            inputs=inputs,
            stream=True
        )
    except writerai.APIError as error:
        print(f"Error: {error}")
        sys.exit(1)

def stream_writerai_chat_completions(
    prompt: str,
    api_key: str,
    model: str,
    max_tokens: str,
    temperature: str,
    system_prompt: str,
    graphs_description: Optional[str],
    graph_ids: Optional[List[str]],
    image_ids: Optional[List[str]],
    response_schema: Optional[str]
) -> writerai.Stream:
    """Stream chat completions from the Writerai API."""
    if writerai is None:
        print("Error: Writerai Python package is not installed.")
        print("Please install by running `pip install writerai'.")
        sys.exit(1)

    if api_key == "NOT SET":
        print("Error: Writerai API key not set.")
        print(
            'Add (setq gpt-writerai-key "sk-Aes.....AV8qzL") to your Emacs init.el file.'
        )
        sys.exit(1)

    client = writerai.Writer(api_key=api_key)

    messages = [{"role": "system", "content": system_prompt}]
    pattern = re.compile(
        r"^(User|Assistant):(.+?)(?=\n(?:User|Assistant):|\Z)", re.MULTILINE | re.DOTALL
    )
    # print(prompt)

    matches = pattern.finditer(prompt)
    for match in matches:
        role = match.group(1).lower()
        content = match.group(2).strip()
        messages.append(
            {"role": role, "content": content}
        )

    tool_calls = []
    try:
        with open("/tmp/tool-calls.json", 'r') as file:
            data = json.load(file)
            tool_calls = data
        os.remove("/tmp/tool-calls.json")
    # except FileNotFoundError:
    #     print("The file /tmp/tool-calls.json was not found.")
    # except json.JSONDecodeError as e:
    #     print(f"Failed to decode JSON: {e}")
    except Exception:
        # print(f"An error occurred: {e}")
        ()
    for call in tool_calls:
        messages.append(call)

    # print("---")

    # print(messages)

    try:
        tools = []
        try:
            with open("/tmp/writer-ai-model-inputs.json", 'r') as file:
                data = json.load(file)
                # print(data["function-tools"])
                tools = data["function-tools"] or []
                os.remove("/tmp/writer-ai-model-inputs.json")
        # except FileNotFoundError:
        #     print("The file /tmp/writer-ai-model-inputs.json was not found.")
        # except json.JSONDecodeError as e:
        #     print(f"Failed to decode JSON: {e}")
        except Exception:
            # print(f"An error occurred: {e}")
            ()
        if graph_ids is not None and model.startswith("palmyra-x"):
            tools.append(
                {
                    "type": "graph",
                    "function": {
                        "description": graphs_description
                        or "a graph with relevant information",
                        "graph_ids": graph_ids,
                        "subqueries": False,
                    },
                })
        if image_ids is not None and model.startswith("palmyra-x"):
            tools.append(
                {
                    "type": "vision",
                    "function": {
                        "model": "palmyra-vision",
                        "variables": [{'name': str(i), 'file_id': id}
                                      for i, id in enumerate(image_ids)]
                    },
                })
        tools.append({
                "type": "web_search",
                "function": {},
            })
        chat = client.chat.chat(
            model=model,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            stream=True,
            tool_choice="auto" if tools else None,
            tools=tools if tools else None,
            response_format=writerai.types.chat_chat_params.ResponseFormat(
                type='json_schema',
                json_schema=response_schema)
            if response_schema else None
        )
        return chat
    except writerai.APIError as error:
        print(f"Error: {error}")
        sys.exit(1)


def print_and_collect_completions(stream, api_type: APIType) -> str:
    """Print and collect completions from the stream."""
    completion_text = ""
    if api_type == "openai":
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                completion_text += text
    elif api_type == "anthropic":
        for chunk in stream:
            if chunk.type == "content_block_delta":
                text = chunk.delta.text
                print(text, end="", flush=True)
                completion_text += text
    elif api_type == "writerai":
        streaming_content = ""
        function_calls = []

        for chunk in stream:
            choice = chunk.choices[0]

            if choice.delta:
                # Check for tool calls
                if choice.delta.tool_calls:
                    for tool_call in choice.delta.tool_calls:
                        if tool_call.id:
                            # Append an empty dictionary to the function_calls list with the tool call ID
                            function_calls.append(
                                {"name": "", "arguments": "", "call_id": tool_call.id}
                            )
                        if tool_call.function:
                            # Append function name and arguments to the last dictionary in the function_calls list
                            function_calls[-1]["name"] += (
                                tool_call.function.name
                                if tool_call.function.name
                                else ""
                            )
                            function_calls[-1]["arguments"] += (
                                tool_call.function.arguments
                                if tool_call.function.arguments
                                else ""
                            )
                # Handle non-tool-call content
                elif choice.delta.content:
                    text = choice.delta.content
                    print(text, end="", flush=True)
                    streaming_content += text

                # A finish reason of tool_calls means the model has finished deciding which tools to call
                elif choice.finish_reason == "tool_calls":
                    with open("/tmp/tools-arguments.json", "w") as outfile:
                        outfile.write(json.dumps(function_calls))
        for chunk in stream:
            # print(chunk)
            try:
                text = chunk.choices[0].delta.content
            except:
                text = chunk.delta.content
            if text:
                print(text, end="", flush=True)
                completion_text += text
    else:
        raise ValueError(f"Unsupported API type '{api_type}'")

    return completion_text


def write_to_jsonl(prompt: str, completion: str, path: Path) -> None:
    """Write the prompt and completion to a jsonl file."""
    if jsonlines is None:
        return
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8"):
            pass  # Create the file
    try:
        with jsonlines.open(path, mode="a") as writer:
            writer.write({"prompt": prompt, "completion": completion})
    except IOError as error:
        print(f"Error: {error}")
        sys.exit(1)


def main() -> None:
    """
    Main function to read a prompt from a file, generate completions
    using the specified API, and save the completions to a JSONL file.
    """
    args = parse_args()
    with open(args.prompt_file, "r") as prompt_file:
        prompt = prompt_file.read()

    if args.api_type == "openai":
        stream = stream_openai_chat_completions(
            prompt,
            args.api_key,
            args.model,
            args.max_tokens,
            args.temperature,
            args.system_prompt,
        )
    elif args.api_type == "anthropic":
        stream = stream_anthropic_chat_completions(
            prompt, args.api_key, args.model, args.max_tokens, args.temperature
        )
    elif args.api_type == "writerai":
        if args.application_id is None:
            stream = stream_writerai_chat_completions(
                prompt,
                args.api_key,
                args.model,
                args.max_tokens,
                args.temperature,
                args.system_prompt,
                args.graphs_description,
                args.graph_ids,
                args.image_ids,
                args.response_schema
            )
        else:
            stream = stream_writerai_app_completions(
                args.application_id,
                args.inputs,
                args.api_key
            )
    else:
        print(f"Error: Unsupported API type '{args.api_type}'")
        sys.exit(1)

    completion_text = print_and_collect_completions(stream, args.api_type)
    file_name = Path.home() / ".emacs_prompts_completions.jsonl"
    write_to_jsonl(prompt, completion_text, file_name)


if __name__ == "__main__":
    main()
