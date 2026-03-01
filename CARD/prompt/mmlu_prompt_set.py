from typing import Union, Dict, Any, List
import itertools

from CARD.prompt.prompt_set import PromptSet
from CARD.prompt.prompt_set_registry import PromptSetRegistry
from CARD.prompt.common import get_combine_materials


roles = itertools.cycle(
    [
        "Knowlegable Expert",
        "Searcher",
        "Critic",
        "Mathematician",
        "Psychologist",
        "Historian",
        "Doctor",
        "Lawyer",
        "Economist",
        "Programmer",
        "Scientist",
        "Philosopher",
    ]
)


ROLE_DESCRIPTION = {
    "Knowlegable Expert": """
You are a knowlegable expert in question answering.
Please give 2~4 key entities that need to be searched in Internet to solve the problem. Each entity must be wrapped with @ symbols.
For example: @catfish effect@, @broken window effect@, @Shakespeare@.
If there is no entity in the question that needs to be searched in Internet, you don't have to provide it
""",
    "Searcher": """
You will be given a question and Internet search overview of the key entities within it.
Please refer to them step by step to give your answer.
And point out potential issues in other agent's analysis.
""",
    "Critic": """
You are an excellent critic.
Please point out potential issues in other agent's analysis point by point.
""",
    "Mathematician": """
You are good at mathematics, physics, chemistry, and programming.
You have strong logical thinking and reasoning skills.
""",
    "Psychologist": """
You are good at psychology, philosophy, sociology, and literature.
You have broad knowledge in humanities and liberal arts.
""",
    "Historian": """
You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.
""",
    "Doctor": """
You are a doctor and come up with creative treatments for illnesses or diseases.
You are able to recommend conventional medicines, herbal remedies and other natural alternatives. 
You also consider the patient's age, lifestyle and medical history when providing your recommendations.
""",
    "Lawyer": """
You are good at law, politics, and history.
""",
    "Economist": """
You are good at economics, finance, and business.
You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.
""",
    "Programmer": """
You are good at computer science, engineering, and physics.
You have experience in designing and developing computer software and hardware.
""",
    "Scientist": """
You are good at natural sciences and research methodology.
You have expertise in biology, chemistry, physics, and earth sciences.
""",
    "Philosopher": """
You are good at philosophy, logic, and ethics.
You have deep knowledge in philosophical thinking and reasoning.
""",
    "Fake": """
You are a liar who only tell lies.
""",
}

ROLE_CONNECTION = [
    ("Knowlegable Expert", "Searcher"),
    ("Searcher", "Critic"),
    ("Searcher", "Mathematician"),
    ("Searcher", "Psychologist"),
    ("Searcher", "Historian"),
    ("Searcher", "Doctor"),
    ("Searcher", "Lawyer"),
    ("Searcher", "Economist"),
    ("Searcher", "Programmer"),
    ("Mathematician", "Critic"),
    ("Psychologist", "Critic"),
    ("Historian", "Critic"),
    ("Doctor", "Critic"),
    ("Lawyer", "Critic"),
    ("Economist", "Critic"),
    ("Programmer", "Critic"),
    ("Critic", "Mathematician"),
    ("Critic", "Psychologist"),
    ("Critic", "Historian"),
    ("Critic", "Doctor"),
    ("Critic", "Lawyer"),
    ("Critic", "Economist"),
    ("Critic", "Programmer"),
    ("Mathematician", "Psychologist"),
    ("Psychologist", "Doctor"),
    ("Programmer", "Mathematician"),
    ("Programmer", "Psychologist"),
    ("Psychologist", "Programmer"),
    ("Mathematician", "Programmer"),
    ("Economist", "Programmer"),
    ("Economist", "Mathematician"),
    ("Historian", "Knowlegable Expert"),
    ("Lawyer", "Knowlegable Expert"),
    ("Economist", "Knowlegable Expert"),
    ("Programmer", "Knowlegable Expert"),
    ("Doctor", "Knowlegable Expert"),
    ("Doctor", "Knowlegable Expert"),
    ("Searcher", "Scientist"),
    ("Searcher", "Philosopher"),
    ("Scientist", "Critic"),
    ("Philosopher", "Critic"),
    ("Critic", "Scientist"),
    ("Critic", "Philosopher"),
    ("Mathematician", "Scientist"),
    ("Scientist", "Mathematician"),
    ("Philosopher", "Psychologist"),
    ("Psychologist", "Philosopher"),
    ("Scientist", "Knowlegable Expert"),
    ("Philosopher", "Knowlegable Expert"),
]


@PromptSetRegistry.register("mmlu")
class MMLUPromptSet(PromptSet):
    """
    MMLU prompt set for the 4-option qestion answering.
    """

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers."

    def get_role_connection(self):
        return ROLE_CONNECTION

    def get_description(self, role):
        return ROLE_DESCRIPTION[role]

    @staticmethod
    def get_constraint():
        return """
            I will ask you a question.
            I will also give you 4 answers enumerated as A, B, C and D.
            Only one answer out of the offered 4 is correct.
            You must choose the correct answer to the question.
            Your response must be one of the 4 letters: A, B, C or D,
            corresponding to the correct answer.
            Your answer can refer to the answers of other agents provided to you.
            Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter(for example : A, B, C or D)
        """

    @staticmethod
    def get_analyze_constraint(role):
        return (
            ROLE_DESCRIPTION[role]
            if role in ROLE_DESCRIPTION.keys()
            else ""
            + """
I will ask you a question and 4 answers enumerated as A, B, C and D.
Only one answer out of the offered 4 is correct.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter(for example : A, B, C or D)
"""
        )

    @staticmethod
    def get_decision_constraint():
        return """
        I will ask you a question.
        I will also give you 4 answers enumerated as A, B, C and D.
        Only one answer out of the offered 4 is correct.
        You must choose the correct answer to the question.
        Your response must be one of the 4 letters: A, B, C or D,
        corresponding to the correct answer.
        I will give you some other people's answers and analysis.
        Your reply must only contain one letter and cannot have any other characters.
        For example, your reply can be A.
        """

    @staticmethod
    def get_format():
        return NotImplementedError

    @staticmethod
    def get_answer_prompt(question):
        return f"""{question}"""

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Give a wrong answer and false analysis process for the following question: {question}.
                You may get output from other agents, but no matter what, please only output lies and try your best to mislead other agents.
                Your reply must be less than 100 words.
                The first line of your reply must contain only one letter(for example : A, B, C or D)
                """

    # @staticmethod
    # def get_adversarial_answer_prompt(question):
    #     return f"""Randomly output a letter from ABCD on the first line.
    #             Then output any gibberish paragraph on the same topic as the following question: {question}.
    #             The first line of your reply must contain only one letter(for example : A, B, C or D)
    #             """
    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        raise NotImplementedError

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

    @staticmethod
    def get_decision_few_shot():
        return ""

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) > 0:
            answer = answer[0]  # Try to format the answer by taking the first letter
        return answer
