from typing import Optional

import langchain
from dotenv import load_dotenv
from langchain import PromptTemplate, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import ValidationError
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor

from .prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, REPHRASE_TEMPLATE, SUFFIX
from .tools import make_tools
from .transcript import TranscriptCallbackHandler

# deepseek llm
def _make_llm(model, temp, api_key, streaming: bool = False, api_base: Optional[str] = None):
    if model.startswith(("gpt-3.5-turbo", "gpt-4", "deepseek-")):
        model_kwargs = {}

        # ChemCrow使用文本ReAct格式调用工具。
        # 关闭DeepSeek思考模式可以减少旧版LangChain解析异常。
        if model.startswith("deepseek-"):
            model_kwargs["thinking"] = {"type": "disabled"}

        llm = langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,
            openai_api_base=api_base,
            model_kwargs=model_kwargs,

            # 避免tiktoken不认识DeepSeek模型名
            tiktoken_model_name="gpt-3.5-turbo",
        )

    elif model.startswith("text-"):
        llm = langchain.OpenAI(
            temperature=temp,
            model_name=model,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,
            openai_api_base=api_base,
        )

    else:
        raise ValueError(f"Invalid model name: {model}")

    return llm

class ChemCrow:
    def __init__(
        self,
        tools=None,

        model="deepseek-flash",
        tools_model="deepseek-flash",

        temp=0.1,
        max_iterations=40,
        verbose=True,
        streaming: bool = True,
        openai_api_key: Optional[str] = None,

        api_base: Optional[str] = None,

        api_keys: Optional[dict] = None,
        local_rxn: bool = False,
        save_transcripts: bool = True,
        result_dir: Optional[str] = None,
        enzyme_planner_config: Optional[dict] = None,
    ):
        """Initialize ChemCrow agent."""
        load_dotenv()
        try:

            self.llm = _make_llm(
                model,
                temp,
                openai_api_key,
                streaming=streaming,
                api_base=api_base,
            )

        except ValidationError:

            raise ValueError("Invalid LLM API key or API base URL")

        if tools is None:
            api_keys = dict(api_keys or {})
            api_keys["OPENAI_API_KEY"] = openai_api_key

            tools_llm = _make_llm(
                tools_model,
                temp,
                openai_api_key,
                streaming=streaming,
                api_base=api_base,
            )

            tools = make_tools(
                tools_llm,
                api_keys=api_keys,
                local_rxn=local_rxn,
                verbose=verbose,
                enzyme_planner_config=enzyme_planner_config,
            )

        self.transcript_handler = (
            TranscriptCallbackHandler(result_dir=result_dir)
            if save_transcripts
            else None
        )

        # 在达到上限时再让模型根据已有工具结果生成最终答案
        self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=ChatZeroShotAgent.from_llm_and_tools(
                self.llm,
                tools,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
                question_prompt=QUESTION_PROMPT,
            ),
            verbose=verbose,
            max_iterations=max_iterations,
            early_stopping_method="generate",
            return_intermediate_steps=True,
            callbacks=(
                [self.transcript_handler] if self.transcript_handler else None
            ),
        )

        rephrase = PromptTemplate(
            input_variables=["question", "agent_ans"], template=REPHRASE_TEMPLATE
        )

        self.rephrase_chain = chains.LLMChain(prompt=rephrase, llm=self.llm)

    def run(self, prompt):
        outputs = self.agent_executor({"input": prompt})
        return outputs["output"]
