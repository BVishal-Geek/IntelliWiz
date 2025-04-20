from langchain_core.language_models import LLM
from langchain_core.messages import AIMessage
from typing import List, Union
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict
from typing import Any
from langchain_core.tools import BaseTool
from typing import List, Sequence
from langchain_core.runnables import Runnable
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptValue, ChatPromptTemplate

class LlamaLangChainWrapper(LLM, Runnable):
    client: Any  # Define the client attribute
    model: str = "llama3.1-70b"
    temperature: float = 0.7
    max_tokens: int = 1500
    tools: List[BaseTool] = []

    class Config:
        arbitrary_types_allowed = True

    # def _call(self, prompt: str, stop: List[str] = None) -> str:
    #     print("✅ ENTERED _call()")
    #     if isinstance(prompt, list):
    #         print("🧪 FINAL PROMPT TO LLAMA:", prompt)
    #     elif hasattr(prompt, "to_messages"):
    #         prompt = prompt.to_messages()
    #         print("🧪 FINAL PROMPT TO LLAMA:", prompt)

    #     print("🔍 Type of prompt:", type(prompt))
    #     if isinstance(prompt, list):
    #         for i, msg in enumerate(prompt):
    #             print(f"Message {i}, {type(i)}: {msg}, type: {type(msg)}")

    #     try:
    #         messages = []
    #         for msg in prompt:
    #             if msg.type == "system":
    #                 role = "system"
    #             elif msg.type == "human":
    #                 role = "user"
    #             elif msg.type == "ai":
    #                 role = "assistant"
    #             else:
    #                 role = msg.type  # fallback, in case it's a custom tool message or similar

    #             messages.append({
    #                 "role": role,
    #                 "content": msg.content
    #             })
    #         for i, msg in enumerate(messages):
    #             print(f"Message {i}: {msg}, type: {type(msg)}")
    #         print("✅ Successfully converted messages to dict")
    #     except Exception as e:
    #         print("🔥 Error converting messages to dict:", e)
    #         raise

    #     api_request = {
    #         "model": self.model,
    #         "messages": messages,
    #         "temperature": self.temperature,
    #         "max_tokens": self.max_tokens
    #     }
    #     print("✅ Successfully called api_request")

    #     try:
    #         response = self.client.run(api_request)
    #         print("✅ Successfully called Llama API")
    #     except Exception as e:
    #         print("❌ Llama client.run() threw an exception:", e)
    #         raise
    

    #     if response.status_code == 200:
    #         data = response.json()
    #         print("📦 Parsed Llama API response JSON:", data)

    #         try:
    #             return data["choices"][0]["message"]["content"]
    #         except (KeyError, IndexError, TypeError) as e:
    #             raise RuntimeError(f"❌ Failed to parse message content: {e}")
        

    def _call(self, prompt):
        print("📨 Prompt to Llama:", prompt)

        # 🔁 Convert LangChain messages to dict format
        if isinstance(prompt, list):
            prompt = [{"role": msg.type, "content": msg.content} for msg in prompt]

        response = self.client.run({
            "model": self.model,
            "messages": prompt,
            "temperature": 0.7,
            "max_tokens": 1500
        })

        if response.status_code == 200:
            data = response.json()
            print("📦 Raw response:", data)
            content = data["choices"][0]["message"]["content"]
            print("🧠 Llama said:", content)
            return content

        raise RuntimeError(f"❌ Llama API Error: {response.status_code} - {response.text}")
    

    @property    
    def _llm_type(self) -> str:
        return "llama-api"

    # def invoke(self, input: Union[str, dict, ChatPromptValue], **kwargs) -> BaseMessage:
    #     print("✅ ENTERED invoke()")
    #     if isinstance(input, ChatPromptValue):
    #         prompt = input
    #     elif isinstance(input, str):
    #         prompt_template = ChatPromptTemplate.from_template("{input}")
    #         prompt = prompt_template.format_prompt(input=input)
    #     elif isinstance(input, dict):
    #         prompt_template = ChatPromptTemplate.from_messages([  
    #             ("system", "You are a helpful assistant."),
    #             MessagesPlaceholder(variable_name="chat_history"),
    #             ("human", "{input}"),
    #             MessagesPlaceholder(variable_name="agent_scratchpad")
    #         ])
    #         prompt = prompt_template.format_prompt(**input)
    #     else:
    #         raise ValueError("Unsupported input type passed to invoke")

    #     output = self._call(prompt)
    #     print("✅ Llama API response:", type(output))
    #     return AIMessage(content=output)


    def invoke(self, input: Union[str, dict, ChatPromptValue], **kwargs) -> BaseMessage:
        print("✅ ENTERED invoke()")

        if isinstance(input, ChatPromptValue):
            prompt = input.to_messages()

        elif isinstance(input, str):
            prompt_template = ChatPromptTemplate.from_template("{input}")
            prompt = prompt_template.format_prompt(input=input).to_messages()

        elif isinstance(input, dict):
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            prompt = prompt_template.format_prompt(**input).to_messages()

        else:
            raise ValueError("Unsupported input type passed to invoke")

        output = self._call(prompt)
        print("✅ Llama API response:", type(output))
        return AIMessage(content=output)
    
    def bind_tools(self, tools: Sequence[BaseTool]):
        self.tools = tools
        return self