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
Please point out potential issues and wrong in other agent's analysis.
Then give your right answer.
""",
    "Mathematician": """
You are a scientific expert with strong logical thinking and reasoning skills.
You specialize in mathematics, physics, chemistry, medicine, programming and economics.
You excel at:
- Solving complex mathematical problems and proofs
- Analyzing physical systems and phenomena
- Understanding chemical reactions and molecular structures
- Applying medical knowledge and biological principles
- Writing and debugging computer programs
- Modeling economic systems and financial markets
You will participate in a discussion about a question. Give you opinion with your knowledge to help the group solve the question.
Your answers are always precise, logical and well-reasoned.
""",
    "Psychologist": """
You are a humanities expert with broad knowledge in liberal arts.
You specialize in:
- Philosophy and logical reasoning
- Psychology and cognitive science
- Sociology and cultural studies
- Literature and art appreciation
- History and historical analysis
You will participate in a discussion about a question. Give your opinion with your knowledge to help the group solve the question.
Your answers demonstrate deep humanistic insight, critical thinking, and cultural sensitivity.
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
You will participate in a discussion about a question. Give your opinion with your knowledge to help the group solve the question.
""",
    "Analyst": """
You are an excellent Analyst.
Repeat the question and options. 
Based on discussions with other intelligent agents, analyze whether each option is correct or incorrect.
And summarize reasons and explanations.
""",
    "Fake": """
You are a liar who only tell lies.
""",
}

ROLE_CONNECTION = [
    ("Knowlegable Expert", "Mathematician"),
    ("Knowlegable Expert", "Economist"),
    ("Knowlegable Expert", "Lawyer"),
    ("Knowlegable Expert", "Critic"),
    ("Knowlegable Expert", "Psychologist"),
    ("Knowlegable Expert", "Doctor"),
    ("Knowlegable Expert", "Historian"),
    ("Knowlegable Expert", "Programmer"),
    ("Knowlegable Expert", "Critic"),
    ("Mathematician", "Critic"),
    ("Mathematician", "Critic"),
    ("Psychologist", "Critic"),
    ("Critic", "Analyst"),
    ("Mathematician", "Analyst"),
    ("Psychologist", "Analyst"),
    ("Analyst", "Critic"),
    ("Economist", "Lawyer"),
    ("Lawyer", "Critic"),
    ("Critic", "Psychologist"),
    ("Psychologist", "Doctor"),
    ("Doctor", "Historian"),
    ("Historian", "Knowlegable Expert"),
    ("Programmer", "Mathematician"),
    ("Programmer", "Knowlegable Expert"),
    ("Mathematician", "Programmer"),
    ("Programmer", "Economist"),
    ("Economist", "Psychologist"),
    ("Psychologist", "Knowlegable Expert"),
    ("Critic", "Historian"),
    ("Historian", "Economist"),
    ("Lawyer", "Knowlegable Expert"),
    ("Doctor", "Lawyer"),
    ("Mathematician", "Doctor"),
    ("Programmer", "Critic"),
    ("Economist", "Doctor"),
    ("Lawyer", "Critic"),
    ("Psychologist", "Lawyer"),
    ("Historian", "Mathematician"),
    ("Programmer", "Doctor"),
    ("Doctor", "Psychologist"),
    ("Historian", "Programmer"),
    ("Critic", "Economist"),
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
