""" This file contains the agent class for our AI research agent."""
import os
import sys
import anthropic
from MLAgentBench.LLM import complete_text_fast, complete_text
from MLAgentBench.schema import Action
from .agent import Agent
import dspy
from dspy import dsp



format_prompt_dict = {
    "Reflection": "What does the observation mean? If there is an error, what caused the error and how to debug?",
    "Research Plan and Status": "The full high level research plan, with current status and confirmed results of each step briefly annotated. It must only include progress that has been made by previous steps. If there is any update, enclose the new update text in double asterisks **like this**. If there is no update, just copy the previous step Research Plan and Status. The high level plan from the previous step should be fully retained, unless it is intentionally revised.",
    "Fact Check": "List all objective statements in the updates to Research Plan and Status one by one and point out whether it is guessed versus directly confirmed by the previous observation directly above. Performance numbers can only be confirmed by running the code and observing the output.",
    "Thought": "What you are currently doing, what actions to perform and why",
    "Action": "the action to take, should be one of the names of the tools",
    "Action Input": "the input to the action as a valid JSON string",
}


def summary_format(summaries):
    newline = "\n"
    return f"\n{''.join([f'step{i+1}: {summaries[i]}{newline}{newline}' for i in range(len(summaries))])}" 

class SummerizationTask(dspy.Signature):
    '''
    Summarize your actions and observations. Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
    '''
    reflection = dspy.InputField(desc = format_prompt_dict['Reflection'])
    research_plan_and_status = dspy.InputField(desc = format_prompt_dict["Research Plan and Status"])
    fact_check = dspy.InputField(desc=format_prompt_dict["Fact Check"])
    thought = dspy.InputField(desc=format_prompt_dict["Thought"])
    action = dspy.InputField(desc=format_prompt_dict["Action"])
    action_input = dspy.InputField(desc=format_prompt_dict["Action Input"])
    action_ouput = dspy.InputField(desc= "The output of the action input")
    reasoning_summary = dspy.OutputField(desc="Summarize all relevant details of the action objectively")
    action_summary = dspy.OutputField(desc= "Summarize all relevant details of the action objectively")
    observation_summary = dspy.OutputField(desc="Summarize all relevant details in the action output objectively")

class SummarizeResearchLog(dspy.Signature):
    '''
    Concisely summarize and list all relevant information from the research log that will be helpful for future step in this format:
    '''
    steps = dspy.InputField(desc = ["first step", "second steps", "..."], format=summary_format)
    research_log_summary = dspy.OutputField(desc="The summary of all the previous steps. Combine them and highlight relevant facts. List them in dot point format and don't separate just by steps.")


class ResearchTask(dspy.Signature):
    """Perform research based on the given Research Problem. You do not know anything about this problem so far. 
Follow these instructions and do not forget them:
- First, come up with a high level plan based on your understanding of the problem and available tools and record it in the Research Plan and Status. You can revise the plan later.
- Research Plan and Status should well organized and succinctly keep track of 1) high level plan (can be revised), 2) what steps have been done and what steps are in progress, 3) short results and conclusions of each step after it has been performed. 
- Research Plan and Status must only include progress that has been made by previous steps. It should not include results not directly confirmed by the previous observation. 
- Performance numbers and estimates can only be confirmed and included in the status by running the code and observing the output.
- You should come up with a good experiment design that addresses the problem, and whenever applicable, define and measure the baseline performance of the relevant system or model before attempting any improvements.
- Follow the plan and try to achieve the goal as straightforwardly as possible.
- Highlight the supporting experiment results and reasoning before drawing any conclusions. 
- Do not try installing any new packages or libraries.
- If you believe you have solved the problem, you can use the Final Answer action to submit your answer. You can only submit once, so double check that you have achieved the goal before submitting.  """
    tools_available = dspy.InputField(desc="Tools Available To the Agent along with their descriptions")
    research_problem= dspy.InputField(desc="Contains the research problem and constraints")
    relevant_history = dspy.InputField(desc="History of Previous Interactions, when relevant")
    reflection = dspy.OutputField(desc= format_prompt_dict["Reflection"])
    research_plan_and_status = dspy.OutputField(desc= format_prompt_dict["Research Plan and Status"])
    fact_check = dspy.OutputField(desc= format_prompt_dict["Fact Check"])
    though = dspy.OutputField(desc =format_prompt_dict["Thought"])
    action = dspy.OutputField(desc= format_prompt_dict["Action"])
    action_input = dspy.OutputField(desc= format_prompt_dict["Action Input"])



class ResearchDspyMod(dspy.Module):
    def __init__(self, args, env, agent):
        super().__init__()
        self.args = args
        self.env = env
        self.agent = agent
        self.research = dspy.Predict(ResearchTask)
        self.summarize = dspy.Predict(SummerizationTask, lm = dspy.OpenAI( args.fast_llm_name))

    def forward(self, **kwargs):
        last_steps = self.args.max_steps_in_context
        last_observation_step = self.args.max_observation_steps_in_context
        env = self.env
        agent = self.agent
        while not env.is_final() and len(agent.history_steps) < self.args.agent_max_steps:

            curr_step = len(agent.history_steps)

            #### call LLM for next action ###

            ###########################################################
            #     construct prompt for LLM based on previous steps    #
            ###########################################################
            relevant_history = "N/A"
            if curr_step > last_steps:
                if not self.args.no_retrieval:
                    # retrieval action
                    relevant_history = env.execute(Action("Retrieval from Research Log", {"current_plan": ""}))


            for idx in range(max(curr_step - last_steps, 0), curr_step):
                action_string = ""
                action_string = self.print_action(agent.history_steps[idx]["action"], agent.valid_format_entires)
                relevant_history += anthropic.AI_PROMPT + "\n"+ action_string + "\nObservation:"
                if curr_step - idx > last_observation_step:
                    relevant_history += "<Done>\n\n"
                else:
                    try:
                        relevant_history += "\n```\n" + agent.history_steps[idx]["observation"] + "\n```\n\n"
                    except:
                        import pdb; pdb.set_trace()
                
            
            ###############################################
            #     call LLM until the response is valid    #
            ###############################################

            
            
            entries = {}
            

            entries_raw = self.research(tools_available = self.tools_prompt, research_problem = env.research_problem,relevant_history = relevant_history)
            entries["Reflection"] = entries_raw.reflection
            entries["Fact Check"] = entries_raw.fact_check
            entries["Action Input"] = entries_raw.action_input
            entries["Action"] = entries_raw.action
            entries["Research Plan and Status"] = entries_raw.research_plan_and_status
            entries["Thought"] = entries_raw.thought
            log_file = os.path.join(self.log_dir , f"step_{curr_step}_log.log")
            with open(log_file,'a') as file:
                file.write(str(dsp.settings.lm.history[-1]) )

            ########################################################
            #     postprocess LLM output and parse to env actions  #
            ########################################################

            rg = entries["Research Plan and Status"]
            action = entries["Action"].strip()
            raw_action_input = entries["Action Input"]

            new_research_plan_content = rg.strip("```") + "\n\n" 
            entries["Research Plan and Status"] = new_research_plan_content
            entries["Research Plan and Status"]= new_research_plan_content.replace("**", "")

            
            # parse the action input if we can ; other wise just return the original input and wait env to throw back an error
            parsing_error = ""
            try:
                action_input = agent.parse_action_input(raw_action_input, agent.action_infos[action])
            except Exception as e:
                action_input = raw_action_input
                parsing_error = str(e)
                

            with open(os.path.join(agent.log_dir , "main_log"), "a", 1) as f:
                f.write("Step " + str(curr_step) + ":\n")
                f.write(anthropic.AI_PROMPT + "\n" + agent.print_action(entries, agent.valid_format_entires) + "\nObservation:\n")


            ########################################
            #         execute action in env        #
            ########################################

            if type(action_input) == dict:
                observation = env.execute(Action(action, action_input))
            else:
                # parsing failed, give agent parsing error
                usage = ",\n            ".join([f"{k}: [{v}]" for k, v in agent.action_infos[action].usage.items()])
                usage = f"""{{
            {usage}
}}"""
                invalid_action_error = f"The action input for {action} needs to be a valid json with proper entries. You may have missed the comma between entries or used triple quotes (json does not recognizes triple quotes). Please use the correct format and try again:\n{usage}"

                observation = "ActionInputParsingError: "+ parsing_error + "\n" + invalid_action_error


            #######################################################
            #               update history_steps                  #
            #######################################################

            # # if observation is too long, we need to summarize it
            # if len(observation) > 5000:
            #     log_file = os.path.join(self.log_dir , f"step_{curr_step}_summarize_observation_log.log")

            #     print("Observation is too long. Summarizing...", file=sys.stderr)
            #     observation = self.summarize_observation(agent.print_action(entries, agent.valid_format_entires), observation, log_file)

            # agent.history_steps.append({"step_idx": len(env.trace.steps), "action": entries, "observation": observation})

            ## filter out ActionInputParsingError if last step is not action input parsing error
            if not observation.startswith("ActionInputParsingError"):
                agent.history_steps = [step for step in agent.history_steps if not step["observation"].startswith("ActionInputParsingError")]

            with open(os.path.join(agent.log_dir , "main_log"), "a", 1) as f:
                f.write("\n```\n" + agent.history_steps[-1]["observation"] + "\n```\n\n")


            #######################################################
            #      write to research log for retrieval            #
            #######################################################
            if not self.args.no_retrieval:
                summary_of_last_step = "Too long to summarize."
                for _ in range(self.args.max_retries):
                    try:
                        log_file = os.path.join(agent.log_dir , f"step_{curr_step}_summary_log.log")
                        # entries = self.history_steps[-1] ["action"]
                        obs = self.history_steps[-1]["observation"]
                        summary = self.summarize(reflection = entries["Reflection"], research_plan_and_status = entries["Research Plan and Status"],
                                       fact_check = entries["Fact Check"], thought = entries["Thought"],action = entries["Action"], action_input = entries["Action Input"], action_output = obs)
                        summary_of_last_step = f"[REASONING]:{summary.reasoning_summary}\n[ACTION]: {summary.action_summary}\n[OBSERVATION]: {summary.observation_summary}"
                        with open(log_file,'a') as file:
                            file.write(str(dsp.settings.lm.history[-1]) )   
                        break
                    except Exception as e:
                        print(e)
                        print("Trying again.")

                action = "Append Summary to Research Log"
                action_input = { "content": "\n\nStep " + str(curr_step) + ":\n" + summary_of_last_step + "\n"}
                env.execute(Action(action, action_input))

            step_idx = len(env.trace.steps) - 1
            self.save(os.path.join(self.log_dir , f"agent_{step_idx}_{curr_step}.json"))

        if env.is_final():
            return "Finished due to env.is_final() == True"
        else:
            return "Finished due to agent max steps reached"


class DSPyAgent(Agent):
    """This class implements AI research agent with different configurations."""

    def __init__(self, args, env):
        super().__init__(args, env)
        self.valid_format_entires = ["Reflection",  "Research Plan and Status","Fact Check", "Thought","Action", "Action Input"] # use all entries by default
        if args.valid_format_entires:
            self.valid_format_entires = args.valid_format_entires
        llm = dspy.OpenAI(model=args.llm_name)
        dspy.settings.configure(lm=llm)
        self.dspyMod = ResearchDspyMod(args,env,self)
        
        

    def run(self, env):
        return self.dspyMod()
