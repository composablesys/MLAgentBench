""" This file implement the Research agent class using LangChain, which adapts to the MLAgentBench framework."""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import openai
from langchain.llms import OpenAI
import anthropic
from langchain.agents import AgentExecutor
from langchain.agents import initialize_agent
from langchain.agents.tools import Tool
from langchain.chat_models import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import (
    AgentAction,
    AgentFinish
)
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.input import get_color_mapping
from langchain.callbacks import FileCallbackHandler
from langchain.agents.mrkl.output_parser import MRKLOutputParser
from MLAgentBench.schema import Action
from .agent import Agent
from .agent_langchain import AgentExecutorWithState, AnthropicOutputParser, EnvTool
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.llms import OpenAI


initial_prompt = """You are a helpful research assistant. You have access to the following tools:
{tools_prompt}

Research Problem: {task_description}

You do not know anything about this problem so far. 

Follow these instructions and do not forget them:
- First, come up with a high level plan based on your understanding of the problem and available tools and record it in the Research Plan and Status. You can revise the plan later.
- Research Plan and Status should well organized and succinctly keep track of 1) high level plan (can be revised), 2) what steps have been done and what steps are in progress, 3) short results and conclusions of each step after it has been performed. 
- Research Plan and Status must only include progress that has been made by previous steps. It should not include results not directly confirmed by the previous observation. 
- Performance numbers and estimates can only be confirmed and included in the status by running the code and observing the output.
- You should come up with a good experiment design that addresses the problem, and whenever applicable, define and measure the baseline performance of the relevant system or model before attempting any improvements.
- Follow the plan and try to achieve the goal as straightforwardly as possible.
- Highlight the supporting experiment results and reasoning before drawing any conclusions. 
- Do not try installing any new packages or libraries.
- If you believe you have solved the problem, you can use the Final Answer action to submit your answer. You can only submit once, so double check that you have achieved the goal before submitting.

Always respond in this format exactly:
{format_prompt}
Observation: 
```
the result of the action
```
{retrieve_info}
\n {format_instructions}

"""

format_prompt_dict = {
    "Reflection": "What does the observation mean? If there is an error, what caused the error and how to debug?",
    "Research Plan and Status": "The full high level research plan, with current status and confirmed results of each step briefly annotated. It must only include progress that has been made by previous steps. If there is any update, enclose the new update text in double asterisks **like this**. If there is no update, just copy the previous step Research Plan and Status. The high level plan from the previous step should be fully retained, unless it is intentionally revised.",
    "Fact Check": "List all objective statements in the updates to Research Plan and Status one by one and point out whether it is guessed versus directly confirmed by the previous observation directly above. Performance numbers can only be confirmed by running the code and observing the output.",
    "Thought": "What you are currently doing, what actions to perform and why",
    "Action": "the action to take, should be one of the names of the tools",
    "Action Input": "the input to the action as a valid JSON string",
}


execute_tool_prompt = """Execute the {raw_action} using the provided tools; here is the input for the action: {raw_action_input}."""
summarize_observation_prompt = """
Action: {action}
Observation: {observation}
The full observation is too long. 
Summarize the observation concisely in this format:
[Observation]: Summarize all relevant details in the observation objectively

Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
"""

summarize_action_observation_prompt = """
Given your action and the observation: 
[Action]: {action}
[Observation]: {observation}
Summarize your action and the observation in this format:
[Reasoning]: Summarize the reasoning behind the action
[Action]: Summarize all relevant details of the action objectively
[Observation]: Summarize all relevant details in the observation objectively
Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
"""

reload_prompt = """
Here is a summary of relevant actions and observations you have done:
```
{relevant_history}
```
Here are the exact several steps you have done most recently (up to 3 steps):
"""

class LangChainResearchAgent(Agent):
    """ A wrapper class to implement the research agent using LangChain."""

    def __init__(self, args, env):
        super().__init__(args, env)
        self.intermediate_steps = []
        self.iterations = 0
        self.time_elapsed = 0.0
        self.start_time = None
        self.history_steps = list()
        ############################################
        ##=====LangChain Prompt Template==========##
        ############################################
        self.valid_format_entires = ["Reflection", "Research Plan and Status", "Fact Check", "Thought", "Action",
                                     "Action Input"]  # use all entries by default
        if args.valid_format_entires:
            self.valid_format_entires = args.valid_format_entires

        # initial_prompt can be implemented using PromptTemplate in langchain instead of from scratch implementation.
        prompt_template = PromptTemplate.from_template(initial_prompt)
        format_prompt = "\n".join([f"{k}: {format_prompt_dict[k]}" for k in self.valid_format_entires])

        ############################################
        ##=====LangChain init parser entries======##
        ############################################
        self.entries_parser, response_format = self.build_entries()
        self.initial_prompt = prompt_template.format(tools_prompt=self.tools_prompt, task_description = env.research_problem, retrieve_info = '',format_prompt = format_prompt, format_instructions = response_format)



    ################################################
    ##=====Customized LangChain helper tools======##
    ################################################

    ###### tools to use ########
    def build_tools(self, env):
        tools = []
        for tool_name in self.prompt_tool_names:
            tools.append(Tool(
                tool_name,
                EnvTool(self.action_infos[tool_name], env).run,
                self.construct_tool_prompt(tool_name, self.action_infos[tool_name]).replace("{", "{{").replace("}",
                                                                                                               "}}")
            )
            )
        return tools

    ####### parser entries ########
    def build_entries(self):
        Reflection = ResponseSchema(
            name="Reflection",
            description="What does the observation mean? If there is an error, what caused the error and how to debug?",
        )
        ResearchPlanandStatus = ResponseSchema(
            name="Research Plan and Status",
            description="The steps to prepare the recipe, as a unique string.",
        )
        FactCheck = ResponseSchema(
            name="Fact Check",
            description="List all objective statements in the updates to Research Plan and Status one by one and point out whether it is guessed versus directly confirmed by the previous observation directly above. Performance numbers can only be confirmed by running the code and observing the output.",
        )
        Thought = ResponseSchema(
            name="Thought",
            description="What you are currently doing, what actions to perform and why.",
        )
        Action = ResponseSchema(
            name="Fact Check",
            description="the action to take, should be one of the names of the tools.",
        )
        ActionInput = ResponseSchema(
            name="Action Input",
            description="the input to the action as a valid JSON string.",
        )

        output_parser = StructuredOutputParser.from_response_schemas(
            [Reflection, ResearchPlanandStatus, FactCheck, Thought, Action, ActionInput]
        )
        response_format = output_parser.get_format_instructions()
        print(response_format)

        return output_parser, response_format

    def run(self, env):

        ############################################
        ##=====LangChain Init LLM=================##
        ############################################
        if self.args.llm_name.startswith("claude"):
            llm = ChatAnthropic(model=self.args.llm_name,
                                anthropic_api_key=open("claude_api_key.txt").read().strip(), temperature=0.5,
                                max_tokens_to_sample=2000)
            agent_kwargs = {"output_parser": AnthropicOutputParser()}
        elif self.args.llm_name.startswith("gpt-4"):
            llm = OpenAI(model_name = "gpt-4", temperature=0.2, openai_api_key=openai.api_key)
            # test = llm("What is the capital of U.S.?")
            agent_kwargs_action_parser = {"output_parser": AnthropicOutputParser()}
        else:
            # TODO: add support for other agents
            raise NotImplementedError

        last_steps = self.args.max_steps_in_context
        last_observation_step = self.args.max_observation_steps_in_context
        with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
            f.write(self.initial_prompt + "\n")

        while not env.is_final() and len(self.history_steps) < self.args.agent_max_steps:
            curr_step = len(self.history_steps)
            prompt = self.initial_prompt
            #####################################################
            ##=Retrieve Research Memory using LangChain tools=###
            #####################################################
            if (curr_step > self.args.max_steps_in_context) and (not self.args.no_retrieval):
                # relevant_history = env.execute(Action("Retrieval from Research Log", {"current_plan": ""}))
                load_action = "Retrieval from Research Log"
                load_action_input = {"current_plan": ""}
                # env.execute(Action(summarize_action, summarize_action_input))
                tool_prompt_template =  PromptTemplate.from_template(execute_tool_prompt)
                tool_prompt = tool_prompt_template.format(raw_action=load_action, raw_action_input=load_action_input)
                relevant_history = agent.run(tool_prompt)
                reload_prompt_template = PromptTemplate.from_template(reload_prompt)
                reload_info = reload_prompt_template.format(relevant_history = relevant_history)
                prompt += reload_info
            else:
                prompt += "\nNow let's start!\n\n"

            retrieve_info =''
            for idx in range(max(curr_step - last_steps, 0), curr_step):
                action_string = "".join([k + ": " + self.history_steps[idx]["action"][k] for k in self.valid_format_entires])

                retrieve_info += anthropic.AI_PROMPT + "\n"+ action_string + "\nObservation:"
                if curr_step - idx > last_observation_step:
                    retrieve_info += "<Done>\n\n"
                else:
                    try:
                        retrieve_info += "\n```\n" + self.history_steps[idx]["observation"] + "\n```\n\n"
                    except:
                        import pdb; pdb.set_trace()
            # prompt = prompt + retrieve_info

            # initial_prompt can be implemented using PromptTemplate in langchain instead of from scratch implementation.
            prompt_template = PromptTemplate.from_template(initial_prompt)
            format_prompt = "\n".join([f"{k}: {format_prompt_dict[k]}" for k in self.valid_format_entires])

            ############################################
            ##=====LangChain init parser entries======##
            ############################################
            self.entries_parser, response_format = self.build_entries()
            self.initial_prompt = prompt_template.format(tools_prompt=self.tools_prompt,
                                                         task_description=env.research_problem, retrieve_info=retrieve_info,
                                                         format_prompt=format_prompt,
                                                         format_instructions=response_format)

            ###############################################
            ##=======LangChain Init LLM=================###
            ###############################################
            responses = llm(self.initial_prompt)


            #################################################
            ##=====LangChain Parse entries=================##
            #################################################
            parsed_response = self.entries_parser.parse(responses)
            raw_action = parsed_response['Action']
            raw_action_input = parsed_response['Action Input']


            #################################################
            ##=====LangChain agent use tools===============##
            #################################################
            tools = self.build_tools(env)
            tool_prompt_template = PromptTemplate.from_template(execute_tool_prompt)
            tool_prompt = tool_prompt_template.format(raw_action=raw_action, raw_action_input = raw_action_input)
            agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
            observation = agent.run(tool_prompt)
            print(observation)

            #################################################
            ##===LangChain summarize summary observation===##
            #################################################

            summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
            if len(observation) > 5000:

                print("Observation is too long. Summarizing...", file=sys.stderr)
                entires = "".join([k + ": " + parsed_response[k] for k in self.valid_format_entires])
                summarize_prompt_template = PromptTemplate.from_template(summarize_observation_prompt)
                summarize_prompt = summarize_prompt_template.format(action=entires, observation=observation)

                text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000,
                                                               chunk_overlap=500)
                docs_observation = text_splitter.create_documents([summarize_prompt])
                observation = summarize_chain.run(docs_observation)

            #################################################
            ##======LangChain summarize observation========##
            #################################################
            if not self.args.no_retrieval:
                summary_of_last_step = "Too long to summarize."
                entires = "".join([k + ": " + parsed_response[k] for k in self.valid_format_entires])
                summarize_action_observation_prompt_template = PromptTemplate.from_template(summarize_action_observation_prompt)
                summarize_action_obser_prompt = summarize_action_observation_prompt_template.format(action=entires, observation=observation)
                if llm.get_num_tokens(summarize_action_obser_prompt) < 2000:
                    summary_of_last_step = llm(summarize_action_obser_prompt)
                else:
                    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
                    docs_action_observation = text_splitter.create_documents([summarize_action_obser_prompt])
                    summary_of_last_step = summarize_chain.run(docs_action_observation)

                summarize_action = "Append Summary to Research Log"
                summarize_action_input = {"content": "\n\nStep " + str(curr_step) + ":\n" + summary_of_last_step + "\n"}
                # env.execute(Action(summarize_action, summarize_action_input))
                tool_prompt = tool_prompt_template.format(raw_action=summarize_action, raw_action_input=summarize_action_input)
                # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
                _ = agent.run(tool_prompt)

            ##########################################################
            ##==Save main.log, summarized observation, and entries==##
            ##########################################################
            self.history_steps.append({"step_idx": len(env.trace.steps), "action": parsed_response, "observation": observation})

            with open(os.path.join(self.log_dir, "main_log"), "a", 1) as f:
                f.write("Step " + str(curr_step) + ":\n")
                f.write(
                    anthropic.AI_PROMPT + "\n" + entires + "\nObservation:\n" + observation)

            step_idx = len(env.trace.steps) - 1
            self.save(os.path.join(self.log_dir, f"agent_{step_idx}_{curr_step}.json"))


        if env.is_final():
            return "Finished due to env.is_final() == True"
        else:
            return "Finished due to agent max steps reached"
