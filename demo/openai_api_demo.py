import json

from colorama import init, Fore
from openai import OpenAI

from configs import logger
from tool_register import get_tools, dispatch_tool

init(autoreset=True)
client = OpenAI(
    base_url="http://10.84.48.32:22000/v1",
    api_key="xxx"
)

functions = get_tools()


def run_conversation(query: str, stream=False, functions=None, max_retry=5):
    params = dict(model="chatglm3-6b", messages=[{"role": "user", "content": query}], stream=stream)
    if functions:
        params["functions"] = functions
        params["messages"].insert(0, {
            "role": "system",
            "content": "Answer the following questions as best as you can. You have access to the following "
                       "tools:\n" + json.dumps(functions, indent=4, ensure_ascii=False),
        })

    response = client.chat.completions.create(**params)

    for _ in range(max_retry):
        if not stream:
            if response.choices[0].message.function_call:
                function_call = response.choices[0].message.function_call
                logger.info(f"Function Call Response: {function_call.model_dump()}")

                function_args = json.loads(function_call.arguments)
                tool_response = dispatch_tool(function_call.name, function_args)
                logger.info(f"Tool Call Response: {tool_response}")

                params["messages"].append(response.choices[0].message)
                params["messages"].append(
                    {
                        "role": "function",
                        "name": function_call.name,
                        "content": tool_response,  # 调用函数返回结果
                    }
                )
            else:
                reply = response.choices[0].message.content
                logger.info(f"Final Reply: \n{reply}")
                return

        else:
            output = ""
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(Fore.BLUE + content, end="", flush=True)
                output += content

                if chunk.choices[0].finish_reason == "stop":
                    return

                elif chunk.choices[0].finish_reason == "function_call":
                    print("\n")

                    function_call = chunk.choices[0].delta.function_call
                    logger.info(f"Function Call Response: {function_call.model_dump()}")

                    function_args = json.loads(function_call.arguments)
                    tool_response = dispatch_tool(function_call.name, function_args)
                    logger.info(f"Tool Call Response: {tool_response}")

                    params["messages"].append(
                        {
                            "role": "assistant",
                            "content": output
                        }
                    )
                    params["messages"].append(
                        {
                            "role": "function",
                            "name": function_call.name,
                            "content": tool_response,  # 调用函数返回结果
                        }
                    )

                    break
            print("\n")
        response = client.chat.completions.create(**params)


if __name__ == "__main__":
    query = "你是谁"
    print(query + "\n")
    #run_conversation(query, stream=True)

    logger.info("\n=========== next conversation ===========")

    query = "帮我查询北京今天的天气怎么样"
    print(query + "\n")
    run_conversation(query, functions=functions, stream=True)
